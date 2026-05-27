import numpy as np
from spore_clustering import SPORE
from sklearn.cluster import KMeans
from .reparameterized import DBSCAN_, SNN_DBSCAN, DPC
from hdbscan import HDBSCAN
from .randrange_estimator import (
    get_kmeans_ranges, 
    get_hdbscan_ranges, 
    get_dbscan_ranges,
    get_snn_dbscan_ranges,
    get_dpc_ranges
)
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import ParameterSampler
from sklearn.utils import check_random_state
from .genericsearch import genericsearch
import traceback
from scipy.stats import uniform, randint


def randsearch(
    X,
    y,
    algorithms=None,
    n_iter=50,
    random_state=42,
    spore_extra_kwargs={},
    evaluator=adjusted_rand_score,
    internal_eval=False,
    show_progress=False, 
    state=None,
    n_jobs=-1,
):
    if algorithms is None:
        algorithms = ['spore', 'kmeans', 'hdbscan', 'dbscan', 'snn_dbscan', 'dpc']
    
    N, D = X.shape
    k = np.unique(y).shape[0]
    param_distributions = {
        'spore': {
            'class': SPORE,
            'defaults': {
                'nn_kwargs':   ({"random_seed": random_state} if not spore_extra_kwargs.get('exact_knn', True) else None),
                'n_jobs':      n_jobs,
                **spore_extra_kwargs,
            },
            'dist': {
                'z_percentile':         uniform(loc=12.5, scale=100 - 12.5),   
                'retention_rate':       uniform(loc=0.0, scale=0.875 - 0.0), 
                'min_cluster_size':     uniform(loc=0.3, scale=0.7 - 0.3), 
                'expansion_neighbors':  randint(int(np.log2(N)), 3*int(np.log2(N))),
                'nn_kwargs':        [{"random_seed": random_state} if not spore_extra_kwargs.get('exact_knn', True) else None],
                'n_jobs':           [n_jobs],
                **{k: [v] for k, v in spore_extra_kwargs.items()}, 
            }
        },
        'kmeans': {
            'class': KMeans,
            'defaults': {'n_clusters': k, 'init': 'k-means++', 'random_state': random_state, 'n_init': 5},
            'dist': (get_kmeans_ranges(X, n_clusters=k, random_state=random_state) if 'kmeans' in algorithms else None)
        },
        'hdbscan': {
            'class': HDBSCAN,
            'defaults': {'core_dist_n_jobs': n_jobs},
            'dist': ({**get_hdbscan_ranges(X), 'core_dist_n_jobs': [n_jobs]} if 'hdbscan' in algorithms else None)
        },
        'dbscan': {
            'class': DBSCAN_,
            'defaults': {},
            'dist': (get_dbscan_ranges(X) if 'dbscan' in algorithms else None)
        },
        'dpc': {
            'class': DPC,
            'defaults': {'k': k},
            'dist': (get_dpc_ranges(X, k=k) if 'dpc' in algorithms else None)
        },
        'snn_dbscan': {
            'class': SNN_DBSCAN,
            'defaults': {},
            'dist': (get_snn_dbscan_ranges(X) if 'snn_dbscan' in algorithms else None)
        },
    }

    sampler_factory = (lambda dist: ParameterSampler(dist, n_iter=n_iter, random_state=check_random_state(random_state)))
    try:
        return genericsearch(
            parameter_sampler=sampler_factory,
            algorithms=algorithms,
            param_distributions=param_distributions,
            X=X,
            y=y,
            evaluator=evaluator,
            internal_eval=internal_eval,
            show_progress=show_progress,
            state=state
        )
    except Exception as err:
        print(traceback.format_exc())
