"""Provides ways to group similar systems together.
Status: under development.
"""

__author__ = "Mohamed-Rafik Bouguelia"
__license__ = "MIT"
__email__ = "mohamed-rafik.bouguelia@hh.se"

from grand import utils
import pandas as pd, numpy as np

class PeerGrouping:
    '''Construct reference groups
    
    Parameters:
    -----------
    w_ref_group : string
        Time window used to define the reference group, e.g. "7days", "12h" ...
        Possible values for the units can be found in https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.to_timedelta.html
    '''
    
    def __init__(self, w_ref_group):
        self.w_ref_group = w_ref_group
    
    # ===========================================
    def get_target_and_reference(self, uid_test, dt, dffs):
        '''Extracts a test sample and its reference group
        
        Parameters:
        -----------
        uid_test : int
            Index (in dffs) of the test unit. Must be in range(len(dffs)).
        
        dt : datetime
            Current datetime period
            
        dffs : list
            Each element in dffs corresponds to one unit. The length of dffs should be the number of units.
            Each element in dffs is a DataFrame containing the previous data (after features extraction) of the corresponding unit.
        
        Returns:
        --------
            x : array-like, shape (n_features,)
                Test sample extracted from the test unit (dffs[uid_test]) at time dt
            
            Xref : array-like, shape (n_samples, n_features)
                Latest samples in the reference group (other units) over a period of w_ref_group
        '''
        
        utils.validate_reference_grouping_input(uid_test, dt, dffs)
        
        x = dffs[uid_test].loc[dt].values
        Xref = []
        for i, dff in enumerate(dffs):
            if i == uid_test: continue
            Xref += list( dff[dt - pd.to_timedelta(self.w_ref_group) : dt].values )
        
        utils.validate_reference_group(Xref)
        Xref = np.array(Xref)
        return x, Xref
    
    # ===========================================