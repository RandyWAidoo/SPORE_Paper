import pandas as pd
import os
from functools import reduce
import operator
from argparse import ArgumentParser as AP


def integrate(integrated_f, base_f, preserved_algorithms, added_algorithms, excluded_algorithms={}):
    preserve_all = (next(iter(preserved_algorithms)) == "*")
    add_all = (next(iter(added_algorithms)) == "*")
    preserved_algorithms = {p.lower() for p in preserved_algorithms}
    added_algorithms = {a.lower() for a in added_algorithms}
    excluded_algorithms = {a.lower() for a in excluded_algorithms}

    integrated = pd.read_csv(integrated_f).reset_index(drop=True)
    if "Unnamed: 0" in integrated:
        integrated.drop(columns=["Unnamed: 0"], inplace=True)
    original = pd.read_csv(base_f).reset_index(drop=True)
    if "Unnamed: 0" in original:
        original.drop(columns=["Unnamed: 0"], inplace=True)

    records = []
    integrated = integrated[[col for col in integrated.columns if integrated[col].values.dtype != object or not integrated[col].isna().any()]]
    original = original[[col for col in original.columns if original[col].values.dtype != object or not original[col].isna().any()]]
    shared_cols = set(original.columns).intersection(integrated.columns)
    integrated = integrated[[col for col in integrated.columns if col in shared_cols]]
    original = original[[col for col in original.columns if col in shared_cols]]
    groupby = ["Dataset", "N", "D", "D'"]
    
    if preserve_all:
        preserved_algorithms = {a.lower() for a in original["Algorithm"].values}
    if add_all:
        added_algorithms = {a.lower() for a in integrated["Algorithm"].values}
    preserved_algorithms -= excluded_algorithms
    added_algorithms -= excluded_algorithms

    print("| Preserved algorithms:", preserved_algorithms)
    print("| Added algorithms:", added_algorithms)

    for tup, _ in original.groupby(by=groupby, sort=False):
        _original = original[reduce(operator.and_, [ original[groupby[i]] == tup[i] for i in range(len(tup)) ])]
        _integrated = integrated[reduce(operator.and_, [integrated[groupby[i]] == tup[i] for i in range(len(tup)) ])]

        for _, record in _integrated.iterrows():
            if record["Algorithm"].lower() in added_algorithms:
                records.append(record)
        for _, record in _original.iterrows():
            if record["Algorithm"].lower() in preserved_algorithms and record["Algorithm"].lower() not in added_algorithms:
                records.append(record)

    original = pd.DataFrame.from_records(records)
    print(original["Dataset,N,D,D',Algorithm,ARI,NMI,Tuning Time,Runtime".split(',')].to_string(), end="\n\n")
    base = os.path.join("merge", os.path.splitext(os.path.basename(base_f))[0])
    csv = base + ".csv"
    html = base + ".html"
    original.to_csv(csv, index=False)
    original.to_html(html, index=False)
    return set(original["Algorithm"].values), csv

ap = AP()
ap.add_argument('-i', '--integrated', help="Integrated file", type=str)
ap.add_argument('-b', '--base', help="Original file", type=str)
ap.add_argument('-p', '--preserved', help="List of algorithms whose results will be kept/not overwritten", type=str)
ap.add_argument('-a', '--add', help="List of algorithms to add from the new file", type=str)
ap.add_argument('-e', '--exclude', help="List of excluded algorithms", default="")
args = ap.parse_args()

all_integrated = args.integrated.split()
preserved = set(args.preserved.split())
added = set(args.add.split())
excluded = set(args.exclude.split())
idx = 0
curr_base = args.base
while idx < len(all_integrated):
    print("Merging <", curr_base, "> and <", all_integrated[idx], ">", end=":\n")
    preserved, curr_base = integrate(all_integrated[idx], curr_base, preserved, added, excluded)
    idx += 1
