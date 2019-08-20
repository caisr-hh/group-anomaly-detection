from .peer_grouping import PeerGrouping
from .transformer import Transformer
from grand import IndividualAnomalyInductive
from grand.utils import DeviationContext, append_to_df, TestUnitError, NoRefGroupError

import pandas as pd, matplotlib.pylab as plt
from pandas.plotting import register_matplotlib_converters

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

    # TODO nb_features, nb_units can be eliminated and inferred from the first call to predict(..)
    def __init__(self, nb_units, ids_target_units, w_ref_group="7days", w_martingale=15,
                 non_conformity="median", k=20, dev_threshold=.6, transform=False, w_transform=20):
        self.nb_units = nb_units
        self.ids_target_units = ids_target_units
        self.w_ref_group = w_ref_group
        self.w_martingale = w_martingale
        self.non_conformity = non_conformity
        self.k = k
        self.dev_threshold = dev_threshold
        self.transform = transform
        self.w_transform = w_transform

        self.dfs_original = [ pd.DataFrame( data = [], index = [] ) for _ in range(nb_units) ]
        self.dfs = [ pd.DataFrame( data = [], index = [] ) for _ in range(nb_units) ]
        self.pg = PeerGrouping(self.w_ref_group)
        self.detectors = [ IndividualAnomalyInductive(w_martingale, non_conformity, k, dev_threshold) for _ in range(nb_units) ]
        self.transformers = [Transformer(w=w_transform) for _ in range(nb_units)]
        
    # ===========================================
    # TODO assert len(x_units) == nb_units, or include the name of units with the data ...
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

        self.dfs_original = [append_to_df(self.dfs_original[i], dt, x) for i, x in enumerate(x_units)]

        if self.transform:
            x_units_tr = [transformer.transform(x) for x, transformer in zip(x_units, self.transformers)]
            self.dfs = [append_to_df(self.dfs[i], dt, x) for i, x in enumerate(x_units_tr)]
        else:
            self.dfs = self.dfs_original

        deviations = []
        
        for uid in self.ids_target_units:
            detector = self.detectors[uid]
            
            try:
                x, Xref = self.pg.get_target_and_reference(uid, dt, self.dfs)
                detector.fit(Xref)
                devContext = detector.predict(dt, x)
            except (TestUnitError, NoRefGroupError):
                devContext = DeviationContext(0, 0.5, 0, False) # no deviation by default
            
            deviations.append(devContext)
            
        return deviations
        
    # ===========================================
    def plot_deviations(self):
        '''Plots the anomaly score, deviation level and p-value, over time.
        '''
        register_matplotlib_converters()
        fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2)

        ax0.set_title("Anomaly scores over time")
        ax0.set_xlabel("Time")
        ax0.set_ylabel("Anomaly score")
        for uid in self.ids_target_units:
            T, S = self.detectors[uid].T, self.detectors[uid].S
            ax0.plot(T, S)

        ax1.set_title("Deviation level and p-values over time")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Deviation level")
        for uid in self.ids_target_units:
            T, P, M = self.detectors[uid].T, self.detectors[uid].P, self.detectors[uid].M
            ax1.scatter(T, P, alpha=0.25, marker=".")
            ax1.plot(T, M, label="Unit"+str(uid))
        ax1.axhline(y=self.dev_threshold, color='r', linestyle='--')
        ax1.legend()

        ax2.set_title("Original data")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Feature 0")
        for uid in self.ids_target_units:
            df_original = self.dfs_original[uid]
            ax2.plot(df_original.index, df_original.values[:, 0], marker=".", label="unit {} var {}".format(uid, 0))
        ax2.legend()

        ax3.set_title("Transformed data")
        ax3.set_xlabel("Time")
        ax3.set_ylabel("Feature 0")
        for uid in self.ids_target_units:
            df = self.dfs[uid]
            ax3.plot(df.index, df.values[:, 0], marker=".", label="unit {} var {}".format(uid, 0))
        ax3.legend()

        fig.autofmt_xdate()
        plt.show()
