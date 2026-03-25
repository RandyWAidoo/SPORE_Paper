import numpy as np
from spore_clustering import SPORE
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from hdbscan import HDBSCAN
from .randrange_estimator import get_kmeans_ranges, get_hdbscan_ranges, get_spectral_ranges, get_gmm_ranges
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import ParameterSampler
from sklearn.utils import check_random_state
from .genericsearch import genericsearch
import traceback
from scipy.stats import uniform


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
        algorithms = ['spore', 'kmeans', 'hdbscan', 'spectral', 'gmm']
    
    k = np.unique(y).shape[0]
    param_distributions = {
        'spore-fixed-grid': {
            'class': SPORE,
            'defaults': {
                'nn_kwargs':   ({"random_seed": random_state} if not spore_extra_kwargs.get('exact_knn', True) else None),
                'n_jobs':      n_jobs,
                **spore_extra_kwargs,
            },
            'dist': {
                'neighborhood_percentile': [25, 50, 75, 93.75],   
                'retention_rate':          [0.0, 0.25, 0.5, 0.75], 
                'min_cluster_size':        [.4, .6], 
                'nn_kwargs':               [{"random_seed": random_state} if not spore_extra_kwargs.get('exact_knn', True) else None],
                'n_jobs':                  [n_jobs],
                **{k: [v] for k, v in spore_extra_kwargs.items()}, 
            }
        },
        'spore-rand-grid': {
            'class': SPORE,
            'defaults': {
                'nn_kwargs':   ({"random_seed": random_state} if not spore_extra_kwargs.get('exact_knn', True) else None),
                'n_jobs':      n_jobs,
                **spore_extra_kwargs,
            },
            'dist': {
                'neighborhood_percentile': uniform(loc=12.5, scale=100 - 12.5),   
                'retention_rate':          uniform(loc=0.0, scale=0.875 - 0.0), 
                'min_cluster_size':        uniform(loc=0.3, scale=0.7 - 0.3), 
                'nn_kwargs':               [{"random_seed": random_state} if not spore_extra_kwargs.get('exact_knn', True) else None],
                'n_jobs':                  [n_jobs],
                **{k: [v] for k, v in spore_extra_kwargs.items()}, 
            }
        },
        
        'kmeans': {
            'class': KMeans,
            'defaults': {'n_clusters': k, 'init': 'k-means++', 'random_state': random_state},
            'dist': (get_kmeans_ranges(X, n_clusters=k, random_state=random_state) if 'kmeans' in algorithms else None)
        },
        'hdbscan': {
            'class': HDBSCAN,
            'defaults': {'core_dist_n_jobs': n_jobs},
            'dist': ({**get_hdbscan_ranges(X, rng=check_random_state(random_state)), 'core_dist_n_jobs': [n_jobs]} if 'hdbscan' in algorithms else None)
        },
        'spectral': {
            'class': SpectralClustering,
            'defaults': {'n_clusters': k, 'affinity': 'nearest_neighbors', 'assign_labels': 'kmeans', 'random_state': random_state, 'n_jobs': n_jobs},
            'dist': ({**get_spectral_ranges(X, n_clusters=k, random_state=random_state), 'n_jobs': [n_jobs]} if 'spectral' in algorithms else None)
        },
        'gmm': {
            'class': GaussianMixture,
            'defaults': {'n_components': k, 'init_params': 'k-means++', 'random_state': random_state},
            'dist': (get_gmm_ranges(X, n_clusters=k, random_state=random_state) if 'gmm' in algorithms else None)
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
