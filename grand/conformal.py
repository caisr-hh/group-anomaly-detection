"""Provide functions for conformal anomaly detection.
This module implements functions to compute non-conformity measures, p-values,
and tests for the uniformity of p-values.
"""

__author__ = "Mohamed-Rafik Bouguelia"
__license__ = "MIT"
__email__ = "mohamed-rafik.bouguelia@hh.se"

from grand import utils
from sklearn.neighbors import LocalOutlierFactor
import numpy as np


def get_strangeness(measure="median", k=10):
    utils.validate_measure_str(measure)
    if measure == "median":
        return StrangenessMedian()
    elif measure == "knn":
        return StrangenessKNN(k)
    else:
        return StrangenessLOF(k)


class Strangeness:
    '''Class that wraps StrangenessMedian and StrangenessKNN and StrangenessLOF'''

    def __init__(self):
        self.X = None
        self.scores = None

    def is_fitted(self):
        return self.X is not None

    def fit(self, X):
        utils.validate_fit_input(X)
        self.X = X

    def get(self, x):
        utils.validate_is_fitted(self.is_fitted())
        utils.validate_get_input(x)

    def pvalue(self, score):
        utils.validate_is_fitted(self.is_fitted())
        utils.validate_list_not_empty(self.scores)
        return len([1 for v in self.scores if v > score]) / len(self.scores)


class StrangenessMedian(Strangeness):
    '''Strangeness based on the distance to the median data (or most central pattern)'''

    def __init__(self):
        super().__init__()

    def fit(self, X):
        super().fit(X)
        self.med = np.median(X, axis=0)
        self.scores = [self.get(x) for x in self.X]

    def get(self, x):
        super().get(x)
        dist = np.linalg.norm(x - self.med)
        return dist


class StrangenessKNN(Strangeness):
    '''Strangeness based on the average distance to the k nearest neighbors'''

    def __init__(self, k=10):
        super().__init__()
        self.k = k

    def fit(self, X):
        super().fit(X)
        self.scores = [self.get(x) for x in self.X]

    def get(self, x):
        super().get(x)
        dists = [np.linalg.norm(x - xx) for xx in self.X if not (x is xx)]
        knn_dists = sorted(dists)[:self.k]
        mean_knn_dists = np.mean(knn_dists)
        return mean_knn_dists


class StrangenessLOF(Strangeness):
    '''Strangeness based on the local outlier factor (LOF)'''

    def __init__(self, k=10):
        super().__init__()
        utils.validate_int_higher(k, 0)
        self.k = k
        self.lof = LocalOutlierFactor(n_neighbors=k, novelty=True, contamination="auto")

    def fit(self, X):
        super().fit(X)
        X_ = list(X) + [ X[-1] for _ in range(self.k - len(X)) ]
        self.lof.fit(X_)
        self.scores = -1 * self.lof.negative_outlier_factor_

    def get(self, x):
        super().get(x)
        outlier_score = -1 * self.lof.score_samples([x])[0]
        return outlier_score
