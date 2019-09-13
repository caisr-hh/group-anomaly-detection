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
    '''
    Parameters:
    -----------
    val : float
    values : array-like or list of float values
    
    Returns:
    --------
    pval : float, in [0, 1]
        p-value representing the proportion of values in that are larger than the given value (val).
    '''
    
    utils.validate_list_not_empty(values)
    return np.mean([1. if v > val else 0. for v in values])


# ==============================================
def martingale(pvalues):
    '''Additive martingale over the last w steps.
    
    Parameters:
    -----------
    pvalues : list
        List of p-values, each value is in [0, 1]
    
    Returns:
    --------
    normalized_one_sided_mart : float, in [0, 1]
        Deviation level. A normalized version of the current martingale value.
    '''
    
    utils.validate_list_not_empty(pvalues)
    utils.validate_all_in_range(pvalues, (0, 1))
    
    betting = lambda p: -p + .5
    normalized_mart = np.sum([ betting(p) for p in pvalues ]) / (.5 * len(pvalues))
    normalized_one_sided_mart = max( normalized_mart, 0 )
    return normalized_one_sided_mart
    
# ==============================================
class StrangenessMedian:
    '''Strangeness based on the distance to the median data (or most central pattern)
    
    Attributes:
    -----------
    med : array-like, shape (n_features,)
    '''
    
    def __init__(self):
        self.med = None
        self.X = None
    
    def is_fitted(self):
        return self.med is not None
        
    def fit(self, X):
        '''Computes the median data in X
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
        '''
        
        self.med = np.median(X, axis=0)
        self.X = X
        return self
    
    def get_fit_scores(self):
        return [ self.get(xx) for xx in self.X ]
        
    def get(self, x):
        '''Computes the strangeness of x with respect to X
        
        Parameters:
        -----------
        x : array-like, shape (n_features,)
            Sample for which the strangeness is computed.
            
        Returns:
        --------
        dist : float
            Euclidean distance between x and med
        '''
        
        dist = np.linalg.norm(x - self.med)
        return dist
    
# ==============================================
class StrangenessKNN:
    '''Strangeness based on the distance to the median data (or most central pattern)
    
    Attributes:
    -----------
    k : int
        Parameter to find the distance to the k-nearest-neighbours
    
    X : array-like, shape (n_samples, n_features)
    '''
    
    def __init__(self, k = 10):
        utils.validate_int_higher(k, 0)
        
        self.k = k
        self.X = None
    
    def is_fitted(self):
        return self.X is not None
        
    def fit(self, X):
        '''Keeps reference to X for computing knn
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
        '''
        
        self.X = X
        return self
    
    def get_fit_scores(self):
        return [ self.get(xx) for xx in self.X ]
        
    def get(self, x):
        '''Computes the strangeness of x with respect to X
        
        Parameters:
        -----------
        x : array-like, shape (n_features,)
            Sample for which the strangeness is computed.
            
        Returns:
        --------
        mean_knn_dists : float
            Average distance between x and its k-nearest-neighbours from X
        '''
        
        dists = [np.linalg.norm(x - xx) for xx in self.X]
        knn_dists = sorted(dists)[:self.k]
        mean_knn_dists = np.mean(knn_dists)
        return mean_knn_dists
        
# ==============================================
class StrangenessLOF:
    '''Strangeness based on the distance to the median data (or most central pattern)
    
    Attributes:
    -----------
    k : int
        Parameter to find the distance to the k-nearest-neighbours
    
    X : array-like, shape (n_samples, n_features)
    '''
    
    def __init__(self, k = 10):
        utils.validate_int_higher(k, 0)
        
        self.k = k
        self.fitted = False
        self.lof = LocalOutlierFactor(n_neighbors=k, novelty=True, contamination="auto")
    
    def is_fitted(self):
        return self.fitted
        
    def fit(self, X):
        '''Keeps reference to X for computing knn
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
        '''
        
        X_ = list(X) + [ X[-1] for _ in range(self.k - len(X)) ]
        self.lof.fit(X_)
        self.fitted = True
        return self
    
    def get_fit_scores(self):
        return -1 * self.lof.negative_outlier_factor_
        
    def get(self, x):
        '''Computes the strangeness of x with respect to X
        
        Parameters:
        -----------
        x : array-like, shape (n_features,)
            Sample for which the strangeness is computed.
            
        Returns:
        --------
        outlier_score : float
            The local outlier factor score
        '''
        
        outlier_score = -1 * self.lof.score_samples([x])[0]
        return outlier_score
        
# ==============================================
class Strangeness:
    '''Class that wraps StrangenessMedian and StrangenessKNN
    '''
    
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
        