from .peer_grouping import PeerGrouping
from cosmo import IndividualAnomalyInductive
from cosmo.utils import DeviationContext, append_to_df, TestUnitError, NoRefGroupError

import pandas as pd

class GroupAnomaly:
    '''Self monitoring for a group of units (machines)
    
    Parameters:
    ----------
    nb_units : int
        Number of units. Must be equal to len(x_units), where x_units is a parameter of the method self.predict
        
    ids_target_units : list
        List of indexes of the target units (to be diagnoised). Each element of the list should be an integer between 0 (included) and nb_units (excluded).
        
    w_ref_group : string
        Time window used to define the reference group, e.g. "7days", "12h" ...
        Possible values for the units can be found in https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.to_timedelta.html
        
    w_martingale : int
        Window used to compute the deviation level based on the last w_martingale samples. 
        
    non_conformity : string
        Strangeness (or non-conformity) measure used to compute the deviation level.
        It must be either "median" or "knn" or "lof"
        
    k : int
        Parameter used for k-nearest neighbours, when non_conformity is set to "knn"
        
    dev_threshold : float
        Threshold in [0,1] on the deviation level
    '''
    
    def __init__(self, nb_units, ids_target_units, w_ref_group="7days", w_martingale=15, non_conformity="median", k=20, dev_threshold=.6):
        self.nb_units = nb_units
        self.ids_target_units = ids_target_units
        self.w_ref_group = w_ref_group
        self.w_martingale = w_martingale
        self.non_conformity = non_conformity
        self.k = k
        self.dev_threshold = dev_threshold
        
        self.dffs = [ pd.DataFrame( data = [], index = [] ) for _ in range(nb_units) ]
        self.pg = PeerGrouping(self.w_ref_group)
        self.detectors = [ IndividualAnomalyInductive(w_martingale, non_conformity, k, dev_threshold) for _ in range(nb_units) ]
        
    # ===========================================
    def predict(self, dt, x_units):
        '''Diagnoise each target unit based on its data x_units[uid] (where uid is in ids_target_units).
        Compute deviation level by comparing the data from the target unit (x_units[uid]) against the reference group.
        
        Parameters:
        -----------
        dt : datetime
            Current datetime period
        
        x_units : array-like, shape (n_units, n_features)
            Each element x_units[i] corresponds to a data-point from the i'th unit at time dt.
            len(x_units) should correspond to the number of units (nb_units).
        
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
        
        self.dffs = [append_to_df(self.dffs[i], dt, x) for i, x in enumerate(x_units)]
        deviations = []
        
        for uid in self.ids_target_units:
            detector = self.detectors[uid]
            
            try:
                x, Xref = self.pg.get_target_and_reference(uid, dt, self.dffs)
                detector.fit(Xref)
                devContext = detector.predict(dt, x)
            except (TestUnitError, NoRefGroupError):
                devContext = DeviationContext(0, 0.5, 0, False) # no deviation by default
            
            deviations.append(devContext)
            
        return deviations
        
    # ===========================================
    def plot_deviations(self):
        '''Plots the p-values and deviation levels over time.
        '''
        for uid in self.ids_target_units:
            self.detectors[uid].plot_deviations()
