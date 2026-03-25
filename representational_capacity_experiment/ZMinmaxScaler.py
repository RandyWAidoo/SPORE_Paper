from sklearn.base import TransformerMixin
from sklearn.preprocessing import MinMaxScaler
import numpy as np

class ZMinmaxScaler(TransformerMixin):
    def __init__(self, z: int = 6):
        self.z = z

    def fit_transform(self, X, y = None, **fit_params):
        X = X.astype(float)
        means = np.mean(X, axis=0)
        stds = np.std(X, axis=0)
        X = np.clip(X, (means - self.z*stds), (means + self.z*stds))
        return MinMaxScaler().fit_transform(X)