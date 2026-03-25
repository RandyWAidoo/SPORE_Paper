from dataclasses import dataclass
from typing import Any, Callable, Tuple, Dict
import multiprocessing as mp
import time
import traceback

@dataclass
class RunOutcome:
    status: str          # "ok" | "timeout" | "error"
    result: Any = None   # final result if ok
    last: Any = None     # last checkpoint written by the worker
    error: str = ""      # repr(exception) if error
    error_trace: str = ""  # traceback.format_exc if error
    duration_s: float = 0.0

def _worker_entry(func: Callable, args: Tuple, kwargs: Dict, state):
    try:
        state["last"] = None
        state["done"] = False
        state["result"] = None
        state["error"] = state["error_trace"] = ""

        res = func(*args, state=state, **kwargs)
        state["result"] = res
        state["done"] = True
    except Exception as e:
        state["error"] = repr(e)
        state["error_trace"] = traceback.format_exc()

def run_with_timeout_and_buffer(
    func: Callable,
    args: Tuple = (),
    kwargs: Dict = None,
    timeout: float = 5.0
) -> RunOutcome:
    if kwargs is None:
        kwargs = {}

    with mp.Manager() as mgr:
        state = mgr.dict()  # shared buffer: state["last"], ["result"], ["done"], ["error"], etc
        state["started"] = False
        p = mp.Process(target=_worker_entry, args=(func, args, kwargs, state))
        p.start()
        while not state["started"]:
            pass
        start = time.perf_counter()
        p.join(timeout)
        end = time.perf_counter()
        duration_s = end - start

        if p.is_alive():
            duration_s = float("inf")
            p.terminate()
            p.join()
            return RunOutcome(
                status="timeout",
                result=None,
                last=state.get("last", None),
                error="",
                error_trace="",
                duration_s=duration_s,
            )

        # Process exited: check for error or success
        err, err_trace = state.get("error", ""), state.get("error_trace", "")
        if err:
            return RunOutcome(
                status="error",
                result=None,
                last=state.get("last", None),
                error=err,
                error_trace=err_trace,
                duration_s=duration_s,
            )

        return RunOutcome(
            status="ok" if state.get("done", False) else "error",
            result=state.get("result", None),
            last=state.get("last", None),
            error="" if state.get("done", False) else "Unknown error",
            error_trace="" if state.get("done", False) else "Unknown error",
            duration_s=duration_s,
        )
