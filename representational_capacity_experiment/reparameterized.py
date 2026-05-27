import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_array
from scipy.spatial.distance import pdist, squareform


class DBSCAN_(BaseEstimator, ClusterMixin):
    def __init__(self, min_samples=5, q_eps=50.0):
        self.min_samples = min_samples
        self.q_eps = q_eps

    def fit(self, X, y=None):
        X = check_array(X)
        if not (0.0 < self.q_eps <= 100.0):
            raise ValueError(f"q_eps must be in (0, 100], got {self.q_eps}")
        nn = NearestNeighbors(n_neighbors=self.min_samples).fit(X)
        knn_dists, _ = nn.kneighbors(X)
        self.eps_ = float(np.percentile(knn_dists[:, -1], self.q_eps))
        self.labels_ = DBSCAN(eps=self.eps_, min_samples=self.min_samples).fit_predict(X)
        return self

    def fit_predict(self, X, y=None):
        return self.fit(X).labels_


class DPC(BaseEstimator, ClusterMixin):
    def __init__(self, q_dc=5.0, k=3):
        self.q_dc = q_dc
        self.k = k

    def fit(self, X, y=None):
        X = check_array(X)
        n = X.shape[0]
        if self.k > n:
            raise ValueError(f"k={self.k} exceeds number of points N={n}")

        dist_mat = squareform(pdist(X))
        self.dc_ = float(np.percentile(dist_mat[np.triu_indices(n, k=1)], self.q_dc))
        self.rho_ = (dist_mat < self.dc_).sum(axis=1) - 1

        order = np.argsort(self.rho_)[::-1]
        delta = np.empty(n)
        nearest_higher = np.full(n, -1, dtype=int)
        delta[order[0]] = dist_mat[order[0]].max()

        for rank in range(1, n):
            i = order[rank]
            higher = order[:rank]
            j = higher[int(np.argmin(dist_mat[i, higher]))]
            delta[i] = dist_mat[i, j]
            nearest_higher[i] = j

        self.delta_ = delta
        self.gamma_ = self.rho_ * delta
        self.cluster_centers_indices_ = np.argsort(self.gamma_)[::-1][:self.k]

        labels = np.full(n, -1, dtype=int)
        for ci, c in enumerate(self.cluster_centers_indices_):
            labels[c] = ci
        if labels[order[0]] == -1:
            labels[order[0]] = labels[self.cluster_centers_indices_[np.argmin(dist_mat[order[0], self.cluster_centers_indices_])]]
        for i in order[1:]:
            if labels[i] == -1:
                labels[i] = labels[nearest_higher[i]]

        self.labels_ = labels
        return self

    def fit_predict(self, X, y=None):
        return self.fit(X).labels_


class SNN_DBSCAN(BaseEstimator, ClusterMixin):
    def __init__(self, k=10, f=0.4, min_samples=5):
        self.k = k
        self.f = f
        self.min_samples = min_samples

    def fit(self, X, y=None):
        X = check_array(X)
        n = X.shape[0]
        self.eps_snn_ = int(np.floor(self.f * self.k))
        self.eps_dist_ = float(self.k - self.eps_snn_)

        nn = NearestNeighbors(n_neighbors=self.k + 1).fit(X)
        _, indices = nn.kneighbors(X)
        neighbor_sets = [set(row[1:]) for row in indices]

        snn_dist = np.full((n, n), float(self.k))
        np.fill_diagonal(snn_dist, 0.0)
        for i in range(n):
            for j in range(i + 1, n):
                d = float(self.k - len(neighbor_sets[i] & neighbor_sets[j]))
                snn_dist[i, j] = snn_dist[j, i] = d

        self.labels_ = DBSCAN(
            eps=self.eps_dist_, min_samples=self.min_samples, metric="precomputed"
        ).fit_predict(snn_dist)
        return self

    def fit_predict(self, X, y=None):
        return self.fit(X).labels_