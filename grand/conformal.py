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


# ==============================================
def pvalue(val, values):
    utils.validate_list_not_empty(values)
    return len([1 for v in values if v > val]) / len(values)


# ==============================================
class StrangenessMedian:
    '''Strangeness based on the distance to the median data (or most central pattern)'''

    def __init__(self):
        self.med = None
        self.X = None
    
    def is_fitted(self):
        return self.med is not None
        
    def fit(self, X):
        self.med = np.median(X, axis=0)
        self.X = X
        return self
    
    def get_fit_scores(self):
        return [ self.get(xx) for xx in self.X ]
        
    def get(self, x):
        dist = np.linalg.norm(x - self.med)
        return dist


# ==============================================
class StrangenessKNN:
    '''Strangeness based on the distance to the median data (or most central pattern)'''
    
    def __init__(self, k = 10):
        utils.validate_int_higher(k, 0)
        
        self.k = k
        self.X = None
    
    def is_fitted(self):
        return self.X is not None
        
    def fit(self, X):
        self.X = X
        return self
    
    def get_fit_scores(self):
        return [ self.get(xx) for xx in self.X ]
        
    def get(self, x):
        dists = [np.linalg.norm(x - xx) for xx in self.X if not (x is xx)]
        knn_dists = sorted(dists)[:self.k]
        mean_knn_dists = np.mean(knn_dists)
        return mean_knn_dists


# ==============================================
class StrangenessLOF:
    '''Strangeness based on the local outlier factor (LOF)'''
    def __init__(self, k = 10):
        utils.validate_int_higher(k, 0)
        
        self.k = k
        self.fitted = False
        self.lof = LocalOutlierFactor(n_neighbors=k, novelty=True, contamination="auto")
    
    def is_fitted(self):
        return self.fitted
        
    def fit(self, X):
        X_ = list(X) + [ X[-1] for _ in range(self.k - len(X)) ]
        self.lof.fit(X_)
        self.fitted = True
        return self
    
    def get_fit_scores(self):
        return -1 * self.lof.negative_outlier_factor_
        
    def get(self, x):
        outlier_score = -1 * self.lof.score_samples([x])[0]
        return outlier_score


# ==============================================
class Strangeness:
    '''Class that wraps StrangenessMedian and StrangenessKNN'''
    
    def __init__(self, measure = "median", k = 10):
        utils.validate_measure_str(measure)
        if measure == "median": self.h = StrangenessMedian()
        elif measure == "knn": self.h = StrangenessKNN(k)
        else: self.h = StrangenessLOF(k)
        
    def is_fitted(self):
        return self.h.is_fitted()
        
    def fit(self, X):
        utils.validate_fit_input(X)
        return self.h.fit(X)
        
    def get_fit_scores(self): # TODO: add tests for this method and for "lof"
        utils.validate_is_fitted(self.is_fitted())
        return self.h.get_fit_scores()
        
    def get(self, x):
        utils.validate_is_fitted(self.is_fitted())
        utils.validate_get_input(x)
        return self.h.get(x)
        