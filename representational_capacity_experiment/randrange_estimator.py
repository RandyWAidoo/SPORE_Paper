import numpy as np
from scipy.stats import randint, uniform


def get_kmeans_ranges(X, n_clusters, random_state):
    return {
        'n_clusters': [n_clusters],
        'init': ['k-means++'], 
        'n_init': [5],
        'random_state': [random_state],
    }

def get_hdbscan_ranges(X):
    n, _ = X.shape
    ms_low, ms_high = 1, int(min(n - 1, 2*np.log2(n)))
    return {
        'min_cluster_size': randint(int(n**0.3), int(n**0.7) + 1),
        'min_samples': randint(ms_low, ms_high + 1),
        'cluster_selection_method': ['leaf', 'eom'],
    }

def get_dbscan_ranges(X):
    N, D = X.shape
    sqN  = np.sqrt(N)
 
    ms_low  = int(min(sqN, D))
    ms_high = int(min(N, 3 * sqN, 3 * D))
    ms_low  = max(1, ms_low)               # must be >= 1
    ms_high = max(ms_low, ms_high)
 
    return {
        'min_samples': randint(ms_low, ms_high + 1),
        'q_eps':       uniform(loc=12.5, scale=100 - 12.5),
    }
 
 
def get_dpc_ranges(X, k):
    return {
        'q_dc': uniform(loc=0, scale=50.0 - 0),
        'k':    [k],
    }
 
 
def get_snn_dbscan_ranges(X):
    N, D = X.shape
    sqN  = np.sqrt(N)
 
    k_low  = max(1, int(np.log2(N)))
    k_high = max(k_low, int(sqN))
 
    ms_low  = int(min(sqN, D))
    ms_high = int(min(N, 3 * sqN, 3 * D))
    ms_low  = max(1, ms_low)
    ms_high = max(ms_low, ms_high)
 
    return {
        'k':           randint(k_low, k_high + 1),
        'f':           uniform(loc=0.0, scale=0.875 - 0.0),
        'min_samples': randint(ms_low, ms_high + 1),
    }

