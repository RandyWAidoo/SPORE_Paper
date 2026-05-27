import pandas as pd
from .utils.relative_performance_util import relative_performance
import sys

stat = input("Stat(One among ARI(default),NMI,Tuning Time,Runtime): ").strip()
if not stat:
    stat = "ARI"

df = pd.read_csv(sys.argv[1])["Dataset,Algorithm,N,D,D',ARI,Std ARI,NMI,Std NMI,Tuning Time,Runtime,Std Runtime".split(',')].reset_index()
df["Dataset"] = df["Dataset"].values + '(' + df["N"].values.astype(str) + ", " + df["D"].values.astype(str) + ", " + df["D'"].values.astype(str) + ')'
df.drop(columns=["N", "D", "D'", "index"], inplace=True)
df.rename(columns={"Dataset": "Dataset(N, D, D')",}, inplace=True)
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
        qef_args = dict(ARI_col = "ARI")
        relative_performance(df, wsrt_kwargs, qef_args=qef_args)
        print()