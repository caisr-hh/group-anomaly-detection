from data import Data
from representation import Representation
from referencegrouping import ReferenceGrouping
from anomaly import Anomaly
from datetime import datetime
import matplotlib.pylab as plt
import pandas as pd

class SelfMonitoring:
    """Self monitoring for a group of units (machines)
    
    Parameters:
    ----------
    rep_type :  string
        Type of data representations to use.
        It must be either "stats" (for simple statistical features) or "hist" (for histogram)
                
    w_ref_group : string
        Time window used to define the reference group, e.g. "7days", "12h" ...
        Possible values for the units can be found in https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.to_timedelta.html
                
    w_martingale : int
        Window used to compute the deviation level based on the last w_martingale samples. 
                
    non_conformity : string
        Strangeness (or non-conformity) measure used to compute the deviation level.
        It must be either "median" or "knn"
                
    k : int
        Parameter used for k-nearest neighbours, when non_conformity is set to "knn"
        
    dev_threshold : float
        Threshold in [0,1] on the deviation level
    """
    
    def __init__(self, rep_type="stats", w_ref_group="7days", w_martingale=15, non_conformity="median", k=50, dev_threshold=.6):
        self.rep_type = rep_type                # Type of data representations: "stats" or "hist"
        self.w_ref_group = w_ref_group          # Time window for the reference group
        self.w_martingale = w_martingale        # Window for computing deviation level
        self.non_conformity = non_conformity    # Strangeness (or non-conformity) measure: "median" or "knn"
        self.k = k                              # Parameter used if non_conformity is "knn"
        self.dev_threshold = dev_threshold      # Threshold in [0,1] on the deviation level (to flag an anomaly)
        
        self.dffs = []
        
        self.rep = Representation(method=self.rep_type)
        self.ref = ReferenceGrouping(self.w_ref_group)
        self.anomaly = Anomaly(self.w_martingale, self.non_conformity, self.k)
    
    # ===========================================
    def diagnoise(self, uid, dt, dfs):
        '''Diagnoise a test unit (identified by uid)
        Compute deviation level by comparing the data from the test unit against the reference group.
        
        Parameters:
        -----------
        uid : int
            Index (in dfs) of the test unit to diagnoise. Must be in range(len(dfs)).
            
        dt : datetime
            Current datetime period
        
        dfs : list
            Each element of the list corresponds to one unit. The length of the list should be the number of units.
            Each element in dfs is a DataFrame containing the data of the corresponding unit over a period of time (starting at dt).
        
        Returns:
        --------
        strangeness : float
            Non-conformity score of the test unit compared to the reference group.
        
        pvalue : float, in [0, 1]
            p-value for the test sample. Represents the proportion of samples in the reference group that are stranger than the test sample.
        
        deviation : float, in [0, 1]
            Scaled deviation level computed based on the martingale method.
        
        is_deviating : boolean
            True if the deviation is above the threshold (dev_threshold)
        '''
        
        x_units = [ self.rep.extract(df.values) for df in dfs ]
        self._add_data_units(dt, x_units)
        
        x, Xref = self.ref.get_target_and_reference(uid, dt, self.dffs)
        strangeness, pvalue, deviation = self.anomaly.deviation(x, Xref)
        is_deviating = deviation > self.dev_threshold
        return (strangeness, pvalue, deviation, is_deviating)
        
    # ===========================================
    def plot_deviation(self):
        '''Plots the p-value and deviation level over time.
        '''
        plt.scatter(range(len(self.anomaly.P)), self.anomaly.P)
        plt.plot(range(len(self.anomaly.M)), self.anomaly.M)
        plt.axhline(y=self.dev_threshold, color='r', linestyle='--')
        plt.show()
        
    # ===========================================
    def _add_data_units(self, dt, x_units):
        '''Method for private use only
        Appends the current data of all units (after representation) to dffs
        '''
        if self.dffs == []:
            self.dffs = [ self._df_append(None, dt, x) for x in x_units ]
        else:
            for i, x in enumerate(x_units):
                self.dffs[i] = self._df_append(self.dffs[i], dt, x)
                
        
    # ===========================================
    def _df_append(self, df, dt, x):
        '''Method for private use only
        Appends a new row to a DataFrame
        '''
        if df is None or len(df) == 0:
            if x != []: return pd.DataFrame( data = [x], index = [dt] )
            else: return pd.DataFrame( data = [], index = [] )
        else:
            if x != []: df.loc[dt] = x
            return df

    