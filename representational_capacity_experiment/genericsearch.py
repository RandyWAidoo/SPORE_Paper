import numpy as np
from tqdm import tqdm
from time import time

def _set_last(state, payload) -> None:
    if state is None:
        return
    try:
        state["last"] = payload.copy()
    except Exception:
        pass

def genericsearch(
    parameter_sampler,
    algorithms,
    param_distributions,
    X,
    y,
    evaluator,
    internal_eval=False,
    show_progress=False,
    state=None
):
    if state is not None:
        state["started"] = True

    results = {"anytime": {algo: [] for algo in algorithms}, "everytime": {algo: [] for algo in algorithms}}
    for idx, algo in enumerate(algorithms, start=1):
        if algo not in param_distributions:
            raise ValueError(f"Unknown algorithm: {algo!r}")

        Cls   = param_distributions[algo]['class']
        defaults = param_distributions[algo]['defaults']

        all_scores = []
        all_params = []
        all_labels = []
        max_i = 0
        min_runtime = 0

        # evaluate default parameters
        runtime = time()
        default_labels = Cls(**defaults).fit_predict(X)
        min_runtime = runtime = time() - runtime
        score_default = (
            evaluator(y, default_labels)
            if not internal_eval
            else (evaluator(X, default_labels) if len(np.unique(default_labels)) > 1 else -np.inf)
        )
        all_scores.append(score_default)
        all_params.append(defaults)
        all_labels.append(default_labels)
        results[algo] = {
            'best_score':  all_scores[max_i],
            'best_params': all_params[max_i],
            'best_labels': all_labels[max_i]
        }
        results["anytime"][algo].append(results[algo]["best_score"])
        results["everytime"][algo].append(score_default)
        _set_last(state, results)

        # iterate over sampled parameter sets
        dist  = param_distributions[algo]['dist']
        sampler = parameter_sampler(dist)
        if show_progress:
            sampler = tqdm(sampler, desc=f"{idx}/{len(algorithms)}: {algo}")

        for (i, params) in enumerate(sampler, start=1):
            runtime = time()
            model = Cls(**params)
            labels = model.fit_predict(X)
            runtime = time() - runtime

            score = (
                evaluator(y, labels)
                if not internal_eval
                else (evaluator(X, labels) if len(np.unique(labels)) > 1 else -np.inf)
            )

            all_scores.append(score)
            all_params.append(params)
            all_labels.append(labels)
            if (score, -runtime) > (all_scores[max_i], -min_runtime):
                max_i = i
                results[algo] = {
                    'best_score':  all_scores[max_i],
                    'best_params': all_params[max_i],
                    'best_labels': all_labels[max_i]
                } 
                min_runtime = runtime
            results["anytime"][algo].append(results[algo]["best_score"])
            results["everytime"][algo].append(score)
            _set_last(state, results)
        _set_last(state, results)

        if show_progress:
            print(f"→ {algo}: score={all_scores[max_i]:.4f}, params={all_params[max_i]}")

    _set_last(state, results) # Guarantee 'last' reflects the final result just before returning
    return results
