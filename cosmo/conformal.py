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
    
    pval = len([v for v in values if v > val]) / len(values)
    return pval

# ==============================================
def martingale(P, w):
    '''Method for private use only.
    Additive martingale over the last w steps.
    TODO: can be computed incrementally (more efficient).
    
    Parameters:
    -----------
    P : list
        List of previous p-values
    
    Returns:
    --------
    normalized_one_sided_mart : float, in [0, 1]
        Deviation level. A normalized version of the current martingale value.
    '''
    
    betting = lambda p: -p + .5
    subP = P[len(P) - w : ] if len(P) > w else P[: w]
    normalized_mart = np.sum([ betting(p) for p in subP ]) / (.5 * w)
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
    
    def is_fitted(self):
        return self.med is not None
        
    def fit(self, X):
        '''Computes the median data in X
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
        '''
        
        self.med = np.median(X, axis=0)
        return self
        
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
class Strangeness:
    '''Class that wraps StrangenessMedian and StrangenessKNN
    '''
    
    def __init__(self, measure = "median", k = 10):
        self.h = StrangenessMedian() if measure == "median" else StrangenessKNN(k)
        
    def is_fitted(self):
        return self.h.is_fitted()
        
    def fit(self, X):
        return self.h.fit(X)
        
    def get(self, x):
        return self.h.get(x)
        