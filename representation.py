import numpy as np

class Representation:
    '''Computes features (i.e. a representation) from several time series
    
    Parameters:
    -----------
    method : string
        "hist" to compute histograms, or "stats" to use simple statistical features such mean, variance etc.
        
    bins : int
        Number of bins to use if method is "hist"
    
    rng : tuple
        Range to use if method is "hist"
    
    normed : bool
        Used if method is "hist". If False, the result will contain the number of samples in each bin. If True, the result is normalized such that the integral over the range is 1.
    '''
    
    def __init__(self, method="hist", bins=50, rng=(0, 15), normed=True):
        self.bins = bins
        self.rng = rng
        self.normed = normed
        
        self._extract = self._hist if method == "hist" else self._stats
    
    # ===========================================
    def extract(self, X):
        '''Extracts features from several time series (presented as columns of X)
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_time_series)
        
        Returns:
        --------
        x : array-like, shape (n_features,)
            Feature-vector containing the features extracted from every column of X
        '''
        x = []
        for j in range(X.shape[1]):
            x += self._extract(X[:, j])
        return np.array(x)
        
    # ===========================================
    def _hist(self, values):
        '''Method for private use only
        Computes a histogram for the given list of values.
        '''
        return np.histogram(values, self.bins, self.rng, density=self.normed)[0].tolist()
    
    # ===========================================
    def _stats(self, values):
        '''Method for private use only
        Computes simple statistical features from the given list of values.
        '''
        return [np.mean(values), np.std(values)]
    