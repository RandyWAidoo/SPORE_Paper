import pandas as pd
from .wilcoxon_srt import wilcoxon_srt

def relative_performance(df: pd.DataFrame, wsrt_args: dict, eps=1e-12, qef_args: dict = {}):
    print(end="\n\n")
    df = df.copy()
    wsrt_args = wsrt_args.copy()

    comparand = wsrt_args["comparand"]
    metric = wsrt_args["metric"]
    index = wsrt_args["index"]
    test_set_column = wsrt_args["test_set_column"]
    
    print(f"\n% of max {metric}")
    percents = df[metric]/df.groupby(index)[metric].transform("max").replace(0, eps) * 100
    df[f"% max {metric} per dataset"] = percents
    print(df.groupby(test_set_column)[f"% max {metric} per dataset"].mean().rename(f"mean % max {metric} per dataset").sort_values(ascending=("time" in metric.lower())).reset_index())

    print(f"\nWilcoxon Signed Rank Test for raw {metric}({comparand} vs Others):")
    print(wilcoxon_srt(df, **wsrt_args))