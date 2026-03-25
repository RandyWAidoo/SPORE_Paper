import warnings
warnings.filterwarnings(module="sklearn", action="ignore")
warnings.filterwarnings(module="umap", action="ignore")
warnings.filterwarnings(
    "ignore",
    message="'force_all_finite' was renamed to 'ensure_all_finite'",
    category=FutureWarning,
    module="sklearn.utils.deprecation",
)


from .evaluate import evaluate_clustering, adjusted_rand_score
def external_evaluator(y_true, y_pred):
    return adjusted_rand_score(y_true, y_pred)


if __name__ == "__main__":
    from types import SimpleNamespace
    import numpy as np
    import time
    import os
    import pandas as pd
    from tqdm import tqdm
    import inspect
    import json
    import pathlib

    from pprint import pprint

    from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
    from .ZMinmaxScaler import ZMinmaxScaler

    from .split_labels import split_labels
    from sklearn.decomposition import PCA

    from .run_with_timeout_and_buffer import run_with_timeout_and_buffer
    from .randsearch import randsearch

    from spore_clustering import SPORE
    from sklearn.cluster import KMeans, SpectralClustering
    from hdbscan import HDBSCAN
    from sklearn.mixture import GaussianMixture

    from .confidence_interval import confidence_interval
    from argparse import ArgumentParser

    import traceback


    AP = ArgumentParser()
    AP.add_argument("--scaler", help="Preprocessing scalar for datasets; one of 'std', 'robust', 'minmax', z(default), or 'none'", default="z", required=False)
    AP.add_argument('--overwrite', help='Overwrite past results in the output directory?', action='store_true', default=False)
    AP.add_argument('--seed', help='The random seed for the experiment', type=int, default=42)
    AP.add_argument('-m', '--max_points', help='The maximum number of points in a dataset', type=int, default=10000)
    AP.add_argument('--n_jobs', help='Max threads used by an algorithm. Default is -1(as many as allowed)', type=int, default=-1)
    AP.add_argument(
        "-l", "--list_path", required=True,
        help="The path to a text file containing newline-separated datasets as they appear in ./datasets/...]"
    )
    AP.add_argument(
        "--area_limit", required=False, type=float, default=float("inf"),
        help="The maximum 'area' of a dataset (NxD') that should be processed. Others will be skipped"
    )
    AP.add_argument('-i', '--include', help="Included algorithms separated by a space", type=str, default="spore-fixed-grid kmeans hdbscan spectral gmm")
    AP.add_argument('--internal_metrics', help='Include internal metrics in evaluation?', action='store_true', default=False)
    AP.add_argument('-o', '--output', help=r"Which directory to save results in", default=None)
    AP.add_argument('-t', '--tuning_time', help="Max number of seconds an algorithm can be tuned", type=float, default=120.0)
    AP.add_argument('--search_trials', help="Number of tuning trials", type=int, default=50)
    AP.add_argument('--var_trials', help="Number of trials to measure variance with the final config", type=int, default=10)

    AP.add_argument('--approx', help="Use approximate knn instead of exact?", action='store_true', default=False)
    AP.add_argument('--mcs', help="A single value for spore's minimum cluster size",  default=None, type=int)
    AP.add_argument('--seeding_order', help="One of 'none', 'random', 'density'(default)", default="density")


    args = AP.parse_args()
    args.scaler = args.scaler.lower()
    args.include = set(args.include.split())


    RANDOM_STATE = args.seed
    VAR_TRIALS = args.var_trials
    SEEDS = [RANDOM_STATE] + [child.generate_state(1)[0] for child in np.random.SeedSequence(RANDOM_STATE).spawn(VAR_TRIALS - 1)]
    SEARCH_TRIALS = args.search_trials
    PRECISION = 4

    records = []
    dsets = [x[:x.rfind(' ')] for x in open(args.list_path).read().split('\n') if x]
    dimensions = [int(x[x.rfind(' '):]) for x in open(args.list_path).read().split('\n') if x]
    methods = {"spore-fixed-grid": SPORE, "spore-rand-grid": SPORE, "kmeans": KMeans, "spectral": SpectralClustering, "gmm": GaussianMixture, "hdbscan": HDBSCAN}
    true_names = {"spore-fixed-grid": "SPORE", "spore-rand-grid": "SPORE", "kmeans": "KMeans", "hdbscan": "HDBSCAN", "spectral": "Spectral", "gmm": "GMM",}

    for x in list(methods.keys()):
        if x not in args.include:
            methods.pop(x, "")


    try:
        print("\nConfig:", vars(args), end="\n\n")
        for dset, dim in zip(dsets, dimensions, strict=True):
            print(f"\033[94m---------------{dset}---------------\033[0m")
            
            path = os.path.join(os.path.join(pathlib.Path(__file__).parent.parent, "datasets", dset))
            df = pd.read_csv(path).dropna()
            too_large = (df.shape[0] > args.max_points)
            if too_large:
                df = df.iloc[np.random.default_rng(RANDOM_STATE).permutation(df.shape[0])[:args.max_points], :]
            for col in df:
                if df[col].dtype == object:
                    df[col] = pd.factorize(df[col].values)[0]
            
            y_candidate_col, y_candidate = None, None
            try:
                y_candidate, y_candidate_col = split_labels(df)[1:]
            except Exception:
                raise ValueError(f"Could not find labels in '{dset}'")
            df.drop(columns=[y_candidate_col], inplace=True)

            for name, METHOD in methods.items():
                scaler = (
                    RobustScaler() if args.scaler == "robust" 
                    else StandardScaler() if args.scaler == "std"
                    else ZMinmaxScaler() if args.scaler == "z"
                    else MinMaxScaler() if args.scaler == "minmax"
                    else None
                )
                orig_X = (scaler.fit_transform(df.values) if scaler is not None else df.values)
            
                X = (
                    orig_X if orig_X.shape[1] <= dim
                    else PCA(n_components=dim, random_state=RANDOM_STATE).fit_transform(orig_X)
                )
                if X.size > args.area_limit:
                    continue

                print(f"{true_names[name]} on '{dset}': ")
                print("Scaler:", scaler)

                spore_extra_kwargs = dict(
                    shuffle_for_hnsw=True, seeding_order=args.seeding_order, shuffle_seed=RANDOM_STATE,
                    exact_knn=(not args.approx),
                )
                if args.mcs is not None:
                    spore_extra_kwargs['min_cluster_size'] = args.mcs
                
                outcome = default_outcome = SimpleNamespace(duration_s=0.0, last={}, error="") 
                anytime_results, everytime_results = [], []
                evaluator = external_evaluator
                outcome = run_with_timeout_and_buffer(
                    func=randsearch,
                    args=(X, y_candidate),
                    kwargs=dict(
                        algorithms=[name], n_iter=SEARCH_TRIALS, random_state=RANDOM_STATE, 
                        spore_extra_kwargs=spore_extra_kwargs,
                        evaluator=evaluator, 
                        show_progress=True, n_jobs=args.n_jobs
                    ),
                    timeout=args.tuning_time
                )
                if outcome.last is not None: 
                    anytime_results = outcome.last["anytime"][name]
                    everytime_results = outcome.last["everytime"][name]
                    outcome.last = outcome.last[name]["best_params"]
                
                best_params, tune_time = (
                    (outcome.last, outcome.duration_s) if outcome.last is not None 
                    else (default_outcome.last, outcome.duration_s)
                )
                all_results = []
                y_s = []
                for i in tqdm(range(VAR_TRIALS), desc="Measuring variance"):
                    run_params = best_params.copy()
                    if name.split('-')[0] == "spore": 
                        run_params.update(spore_extra_kwargs)

                    if name.split('-')[0] == "spore":
                        clust_seed = SEEDS[i]
                        if not run_params.get("exact_knn", True):
                            run_params["shuffle_for_hnsw"] = True
                            run_params["shuffle_seed"] = clust_seed
                            run_params["nn_kwargs"] = {"random_seed": clust_seed}
                    elif "random_state" in inspect.signature(METHOD.__init__).parameters:
                        clust_seed = SEEDS[i]
                        run_params["random_state"] = clust_seed

                    t = time.perf_counter()
                    y_ = METHOD(**run_params).fit_predict(X)
                    t = time.perf_counter() - t
                    
                    results = evaluate_clustering(X, y_candidate, y_, internal_metrics=args.internal_metrics)
                    results["Runtime"] = t
                    y_s.append(y_)
                    
                    results["Run Parameters"] = run_params.copy()
                    all_results.append(results)
                    
                    if t > args.tuning_time:
                        t = np.inf
                        break

                result_df = pd.DataFrame.from_records(all_results)
                core_stats = list(all_results[0].keys())
                core_stats.remove("Run Parameters")
                core_df = result_df[core_stats].copy()
                core_df.reset_index(drop=True, inplace=True)
                record = core_df.mean(axis=0).round(PRECISION)
                record["Tuning Time"] = tune_time

                for stat in core_stats:
                    record[f"Std {stat}"] = (round(core_df[stat].values.std(), PRECISION) if len(y_s) > 1 else np.nan)
                    record[f"Margin {stat}"] = (round(confidence_interval(core_df[stat].values)[0], PRECISION) if len(y_s) > 1 else np.nan)

                pw_aris = []
                for i in range(len(y_s)):
                    for j in range(i + 1, len(y_s)):
                        pw_aris.append(adjusted_rand_score(y_s[i], y_s[j]))
                record["Pairwise-ARI"] = (round(np.mean(pw_aris), PRECISION) if len(y_s) > 1 else np.nan)
                record["Std Pairwise-ARI"] = (round(np.std(pw_aris), PRECISION) if len(y_s) > 1 else np.nan)
                record["Margin Pairwise-ARI"] = (round(confidence_interval(pw_aris)[0], PRECISION) if len(y_s) > 1 else np.nan)
                
                record = {
                    "Dataset": os.path.splitext(dset[dset.find('-') + 1:].strip())[0],
                    "Scaler": str(scaler),
                    "N": orig_X.shape[0],
                    "D": orig_X.shape[1],
                    "D'": X.shape[1],
                    "Algorithm": true_names[name],
                    **record.to_dict(),
                    "Best Parameters": json.dumps(best_params),
                    "Anytime": json.dumps(anytime_results),
                    "Everytime": json.dumps(everytime_results),
                }
                pprint(record, sort_dicts=False)
                print("Error Log:", (None if not outcome.error else outcome.error))
                print("Traceback:", (None if not outcome.error else outcome.error))
                print()
                records.append(record)

            print()
    except (BaseException, KeyboardInterrupt) as err:
        print("Error occurred:")
        print(traceback.format_exc())
        if not args.overwrite or not records or input("Save current results(y/n)?: ").lower() != 'y':
            exit()


    all_results_df = pd.DataFrame.from_records(records)
    print(f"\033[92m---------------Final Results---------------\033[0m")
    print(all_results_df)
    if args.overwrite:
        base_path = os.path.join(args.output, f"exact-knn-{(not args.approx)}_seed-{args.seed}_n-jobs-{args.n_jobs}_tuning-time-{args.tuning_time}s_scaler-{args.scaler}_mcs-{args.mcs}_seeding-order-{args.seeding_order}")
        all_results_df.to_csv(base_path + ".csv", index=False)
        all_results_df.to_html(base_path + ".html", index=False)