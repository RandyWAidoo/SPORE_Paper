import numpy as np
from sklearn.metrics import (
    adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score,
    davies_bouldin_score, calinski_harabasz_score, silhouette_score
)
from clustpy.metrics import unsupervised_clustering_accuracy as acc


def evaluate_clustering(X, y_true, y_pred, internal_metrics=True, external_metrics=True)->dict[str, float]:
    results = {}
    
    if external_metrics:
        results["ARI"] = float(adjusted_rand_score(y_true, y_pred))
        results["NMI"] = float(normalized_mutual_info_score(y_true, y_pred))
        results["ACC"] = float(acc(y_true, y_pred))
        results["AMI"] = float(adjusted_mutual_info_score(y_true, y_pred))
    
    if internal_metrics:
        nclusters = len(np.unique(y_pred))
        results["Silhouette Score"] = (float(silhouette_score(X, y_pred)) if nclusters > 1 else np.nan)
        results["Calinski Harabasz Score"] = (float(calinski_harabasz_score(X, y_pred)) if nclusters > 1 else np.nan)
        results["Davies Bouldin Score"] = (float(davies_bouldin_score(X, y_pred)) if nclusters > 1 else np.nan)
    return results