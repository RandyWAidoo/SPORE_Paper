import pandas as pd
from .wilcoxon_srt import wilcoxon_srt
from .sum_norm_by_index import sum_norm_by_index

def relative_performance(df: pd.DataFrame, wsrt_args: dict, eps=1e-12, qef_args: dict = {}):
    print(end="\n\n")
    df = df.copy()
    wsrt_args = wsrt_args.copy()

    comparand = wsrt_args["comparand"]
    metric = wsrt_args["metric"]
    index = wsrt_args["index"]
    test_set_column = wsrt_args["test_set_column"]

    ARI_col = qef_args.pop("ARI_col", "ARI")
    NMI_col = qef_args.pop("NMI_col", "NMI")
    Runtime_col = qef_args.pop("Runtime_col", "Runtime")
    Tuning_Time_col = qef_args.pop("Tuning Time_col", "Tuning Time")
    df[Tuning_Time_col] = df[Tuning_Time_col].replace(float("inf"), 120.0)

    print(f"\n% of max {metric}")
    percents = df[metric]/df.groupby(index)[metric].transform("max").replace(0, eps) * 100
    df[f"% max {metric} per dataset"] = percents
    print(df.groupby(test_set_column)[f"% max {metric} per dataset"].mean().rename(f"mean % max {metric} per dataset").sort_values(ascending=("time" in metric.lower())).reset_index())

    print(f"\nWilcoxon Signed Rank Test for raw {metric}({comparand} vs Others):")
    print(wilcoxon_srt(df, **wsrt_args))

    print("\nQuality-Efficiency Front:")
    N_D = ["N", "D"] * ("N" in df.columns and "D" in df.columns)
    df[ARI_col] = (df[ARI_col] + 1)/2
    df["sum-norm ARI"] = sum_norm_by_index(df[[index, *N_D, test_set_column, ARI_col]], metric=ARI_col, groupby_cols=[index], aggregation_cols=[test_set_column], eps=eps)[0]["norm"]
    df["sum-norm NMI"] = sum_norm_by_index(df[[index, *N_D, test_set_column, NMI_col]], metric=NMI_col, groupby_cols=[index], aggregation_cols=[test_set_column], eps=eps)[0]["norm"]    
    df["1 - sum-norm Runtime"] = sum_norm_by_index(df[[index, *N_D, test_set_column, Runtime_col]], metric=Runtime_col, lower_is_better=True, groupby_cols=[index], aggregation_cols=[test_set_column], eps=eps)[0]["norm"]
    df["1 - sum-norm Tuning Time"] = sum_norm_by_index(df[[index, *N_D, test_set_column, Tuning_Time_col]], metric=Tuning_Time_col, lower_is_better=True, groupby_cols=[index], aggregation_cols=[test_set_column], eps=eps)[0]["norm"]
    print(df[[
        index, *N_D, test_set_column, 
        "sum-norm ARI", "sum-norm NMI", 
        "1 - sum-norm Runtime", "1 - sum-norm Tuning Time", 
    ]].to_string())
    
    for consumption_name, consumption_col in [("Runtime", "1 - sum-norm Runtime"), ("Tuning Time", "1 - sum-norm Tuning Time")]:
        qef_col = f"Quality-Efficiency Front(Q=(ARI,NMI),C=({consumption_name}))"
        df[qef_col] = ( (df["sum-norm ARI"] * df["sum-norm NMI"])**(1/2) * df[consumption_col] )**(1/2)
        print(f"\nAvg QEF(Q=(ARI,NMI),C=({consumption_name})):")
        print(df.groupby(test_set_column)[qef_col].mean().rename(qef_col).sort_values(ascending=False).reset_index())

        print(f"\nWilcoxon Signed Rank Test for Quality-Efficiency Front(Q=(ARI,NMI),C=({consumption_name})) ({comparand} vs Others):")
        wsrt_args = wsrt_args.copy()
        wsrt_args.pop("metric",  None)
        print(wilcoxon_srt(df, metric=qef_col, **wsrt_args)) 
    
    qef_col = "Quality-Efficiency Front(Q=(ARI,NMI),C=(Runtime, Tuning Time))"
    df[qef_col] = ( (df["sum-norm ARI"] * df["sum-norm NMI"])**(1/2) * (df["1 - sum-norm Runtime"] * df["1 - sum-norm Tuning Time"])**(1/2) )**(1/2)
    print("\nAvg QEF(Q=(ARI,NMI),C=(Runtime, Tuning Time)):")
    print(df.groupby(test_set_column)[qef_col].mean().rename(qef_col).sort_values(ascending=False).reset_index())

    print(f"\nWilcoxon Signed Rank Test for Quality-Efficiency Front(Q=(ARI,NMI),C=(Runtime, Tuning Time)) ({comparand} vs Others):")
    wsrt_args = wsrt_args.copy()
    wsrt_args.pop("metric",  None)
    print(wilcoxon_srt(df, metric=qef_col, **wsrt_args)) 