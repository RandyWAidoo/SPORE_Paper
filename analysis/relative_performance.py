import pandas as pd
from .utils.relative_performance_util import relative_performance
import sys

stat = input("Stat(One among ARI(default),NMI,Tuning Time,Runtime): ").strip()
if not stat:
    stat = "ARI"
stat = "OS-Avg P-Avg " + stat

print('\n' + "-"*50)
print("Understanding The Columns:")
print("- OS stands for Outer Seed. Each outer seed is associated with a full clustering pipeline with its own mean, std deviations, etc., of values across inner trials.")
print("- Pipeline-level statistics are prefixed with 'P-' while metrics computed across outer seeds are prefixed with 'OS-'.")
print("-"*50, end="\n\n")

df = pd.read_csv(sys.argv[1])["Dataset,Algorithm,N,D,D',ARI,Std ARI,NMI,Std NMI,Tuning Time,Runtime,Std Runtime".split(',')].reset_index()
df["Dataset"] = df["Dataset"].values + '(' + df["N"].values.astype(str) + ", " + df["D"].values.astype(str) + ", " + df["D'"].values.astype(str) + ')'
df.drop(columns=["N", "D", "D'", "index"], inplace=True)
df.rename(columns={
    "Dataset": "Dataset(N, D, D')",
    "ARI": "OS-Avg P-Avg ARI",
    "Std ARI": "OS-Avg P-Std ARI",
    "NMI": "OS-Avg P-Avg NMI",
    "Std NMI": "OS-Avg P-Std NMI",
    "Tuning Time": "OS-Avg P-Avg Tuning Time",
    "Runtime": "OS-Avg P-Avg Runtime",
    "Std Runtime": "OS-Avg P-Std Runtime",
}, inplace=True)
print("Data:")
print(df.to_string())
print()

if comparand:= input(f"Algorithm for comparisons(one of {df["Algorithm"].unique().tolist()}): "):
    if comparand not in df["Algorithm"].values:
        print(f"Algorithm {comparand} not found")
    else:
        wsrt_kwargs = dict(
            comparand = comparand,
            metric = stat,
            hb_correction = True,
            need_pivot = True,
            index = "Dataset(N, D, D')",
            test_set_column = "Algorithm",
        ) 
        qef_args = dict(
            ARI_col = "OS-Avg P-Avg ARI",
            NMI_col = "OS-Avg P-Avg NMI",
            Runtime_col = "OS-Avg P-Avg Runtime",
        ) | {"Tuning Time_col": "OS-Avg P-Avg Tuning Time"}
        relative_performance(df, wsrt_kwargs, qef_args=qef_args)
        print()