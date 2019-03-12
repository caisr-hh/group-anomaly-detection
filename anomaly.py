import pandas as pd, numpy as np
from sklearn.preprocessing import StandardScaler
from strangeness import Strangeness

class Anomaly:
    '''Online conformal anomaly detection with the martingale.
    
    Parameters:
    -----------
    w_martingale : int
        Window size (in terms of number of samples) for computing the deviation level using an additive martingale
    
    non_conformity: string
        Non-conformity (strangeness) measure to use: "median" or "knn"
    
    k : int
        Used to compute k-Nearest-Neihbours if non_conformity is "knn"
    '''
    def __init__(self, w_martingale, non_conformity, k):
        self.w_martingale = w_martingale
        self.strg = Strangeness(non_conformity, k)
        
        self.S, self.P, self.M = [], [], []
    
    # =============================================================================
    def deviation(self, x, Xref):
        '''Update the deviation level based on the new test sample x
        
        Parameters:
        -----------
        x : array-like, shape (n_features,)
            New sample from the test unit (for which the strangeness, p-value and deviation-level are computed)
        
        Xref : array-like, shape (n_samples, n_features)
            Samples from units in the reference group
        
        Returns:
        --------
        strangeness : float
            Strangeness of x with respect to samples in Xref
        
        pval : float, in [0, 1]
            p-value that represents the proportion of samples in Xref that are stranger than x.
        
        deviation : float, in [0, 1]
            Normalized martingale-based deviation level updated based-on the last w_martingale steps
        '''
        stg = self.strg.fit(Xref)
        scores = [ stg.get(xx) for xx in Xref ]
        
        strangeness = stg.get(x)
        self.S.append(strangeness)
        
        pval = self._pvalue(strangeness, scores)
        self.P.append(pval)
        
        deviation = self._matingale(self.P)
        self.M.append(deviation)
        
        return strangeness, pval, deviation
    
    # =============================================================================
    def _pvalue(self, val, values):
        '''Method for private use only.
        
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

    # =============================================================================
    def _matingale(self, P):
        '''Method for private use only.
        Additive martingale over the last w_martingale steps.
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
        subP = P[len(P) - self.w_martingale : ] if len(P) > self.w_martingale else P[: self.w_martingale]
        normalized_mart = np.sum([ betting(p) for p in subP ]) / (.5 * self.w_martingale)
        normalized_one_sided_mart = max( normalized_mart, 0 )
        return normalized_one_sided_mart
        
    # =================================================================================
    