import numpy as np
from scipy.stats import randint

def suggest_hdbscan_bands(X, rng=None):
    n, _ = X.shape
    rng = np.random.default_rng() if rng is None else rng

    mcs_candidates = (np.array([0.005, 0.01, 0.02, 0.05]) * n).astype(int)
    mcs_candidates = np.unique(np.clip(mcs_candidates, 2, n - 1))
    mcs_low = int(np.min(mcs_candidates))
    mcs_high = int(np.max(mcs_candidates))
    if mcs_high <= mcs_low:
        mcs_high = min(n - 1, mcs_low + 1)

    ms_low, ms_high = 1, int(min(n - 1, np.log2(n)))

    return {
        "mcs_low": mcs_low,
        "mcs_high": mcs_high,
        "ms_low": ms_low,
        "ms_high": ms_high,
    }

def suggest_n_neighbors(n):
    lbound = int(np.ceil(np.log(n)))
    ubound = int(max(lbound + 1, min( 100, np.ceil(np.sqrt(n)) ))) 
    return {"n_low": lbound, "n_high": ubound}


# --------------------------------------------------------------------------- #
# Distribution builders
# --------------------------------------------------------------------------- #

def get_kmeans_ranges(X, n_clusters, random_state):
    return {
        'n_clusters': [n_clusters],
        'init': ['k-means++'], 
        'random_state': [random_state],
    }


def get_hdbscan_ranges(X, metric="euclidean", rng=None):
    b = suggest_hdbscan_bands(X, rng=rng)
    return {
        'min_cluster_size': randint(b["mcs_low"], b["mcs_high"] + 1),
        'min_samples': randint(b["ms_low"], b["ms_high"] + 1),
        'metric': [metric],
    }


def get_spectral_ranges(X, n_clusters, random_state):
    nneighbors = suggest_n_neighbors(X.shape[0])
    return {
        'n_clusters': [n_clusters],
        'affinity': ['nearest_neighbors'],
        'n_neighbors': randint(nneighbors['n_low'], nneighbors['n_high'] + 1),
        'assign_labels': ['kmeans'],
        'random_state': [random_state],
    }


def get_gmm_ranges(X, n_clusters, random_state):
    return {
        'n_components': [n_clusters],
        'covariance_type': ['full', 'tied', 'diag', 'spherical'],
        'init_params': ['k-means++'],
        'random_state': [random_state],
    }
