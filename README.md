This README covers the necessary information to reproduce or analyze results from the paper.

# Existing results
Merged, cleaned results from the initial experiments and ablations used in the paper are provided as CSV files in `./results`.
Refer to step 6 of the experimental procedure below to perform analysis on the data.

# Experimental procedure
1. Install Python 3.12+ and create and activate a virtual environment.
2. Install Python packages: `pip install -r requirements.txt`.
3. Run the following commands for the experiment and the various ablations:
    - `python -m representational_capacity_experiment.run -o <desired-output-path> --overwrite -l all_datasets.txt -m 35000` 
    - `python -m representational_capacity_experiment.run -o <desired-output-path> --overwrite -l all_datasets.txt -m 35000 -i 'spore-fixed-grid' --approx` 

    - `python -m representational_capacity_experiment.run -o <desired-output-path> --overwrite -l all_datasets.txt -m 35000 --scaler std` 
    - `python -m representational_capacity_experiment.run -o <desired-output-path> --overwrite -l all_datasets.txt -m 35000 -i 'spore-fixed-grid' --approx --scaler std` 
    
    - `python -m representational_capacity_experiment.run -o <desired-output-path> --overwrite -l all_datasets.txt -m 35000 -i 'spore-fixed-grid' --mcs 1` 
    - `python -m representational_capacity_experiment.run -o <desired-output-path> --overwrite -l all_datasets.txt -m 35000 -i 'spore-fixed-grid' --seeding_order random` 
    - `python -m representational_capacity_experiment.run -o <desired-output-path> --overwrite -l all_datasets.txt -m 35000 -i 'spore-rand-grid' --approx` 
4. Rename each SPORE variant in each result CSV such that the names are mutually distinct across results. For example, one may name the approximate variant of SPORE in the Standard Scaler ablation, "SPORE-ANN-Std-Scaler". This way, SPORE records can be meaningfully merged across ablations into 1 file (step 5). The existing results (in `./results`) follow the following naming scheme:
    - SPORE-ENN: SPORE with exact knn and the Z-Clipped Min-Max Scaler (used in the main experiment).
    - SPORE-ANN: SPORE with approximate knn and the Z-Clipped Min-Max Scaler (used in the main experiment).
    - SPORE-ENN-Std-Scaler: SPORE with exact knn and the Standard Scaler.
    - SPORE-ANN-Std-Scaler: SPORE with approximate knn and the Standard Scaler.
    - SPORE-ENN-Rand-Seed: SPORE with exact knn, the Z-Clipped Min-Max Scaler, and random seeding.
    - SPORE-ENN-MCS-1: SPORE with exact knn, the Z-Clipped Min-Max Scaler, and `min_cluster_size` set to 1.
    - SPORE-ANN-Rand-Grid: SPORE with approximate knn, the Z-Clipped Min-Max Scaler, and a grid of randomly sampled values rather than a fixed grid.
5. Merge results across files using `merge_results.py`. For example:

    - `python merge_results.py -b <result-path1> -p 'kmeans hdbscan' -i <result-path2> -a 'SPORE-ENN'` 
    
    will create a file in `./merge` with the same name as the file provided to the `-b` argument(the base file). The merged file will contain algorithm results from (1) the base file, with desired algorithms named by the `-p` argument, and (2) the integrated file(`-i` argument) with desired algorithms named via the `-a` argument. These results will be per-dataset. In the given example, the output file will, for each dataset, list records from K-means, HDBSCAN, and SPORE-ENN.
6. Perform analysis on results via the following commands:
    - Relative performance(Percent-ARI, Wilcoxon tests): `python -m analysis.relative_performance <result_csv_path>`
    - Anytime Performance: `python -m analysis.anytime <result_csv_path>`
