"""A group-based anomaly detection model (for a group of systems).
To detect anomalies, the GroupAnomaly model compares the data from each
target system against a group of other similar systems.
"""

__author__ = "Mohamed-Rafik Bouguelia"
__license__ = "MIT"
__email__ = "mohamed-rafik.bouguelia@hh.se"

from .peer_grouping import PeerGrouping
from .transformer import Transformer
from grand import IndividualAnomalyInductive
from grand.utils import DeviationContext, append_to_df, TestUnitError, NoRefGroupError
from grand import utils
import pandas as pd, matplotlib.pylab as plt, numpy as np
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

    def __init__(self, nb_units, ids_target_units, w_ref_group="7days", w_martingale=15, non_conformity="median", k=20,
                 dev_threshold=.6, transformer="pvalue", w_transform=30, columns=None):
        self.nb_units = nb_units
        self.ids_target_units = ids_target_units
        self.w_ref_group = w_ref_group
        self.w_martingale = w_martingale
        self.non_conformity = non_conformity
        self.k = k
        self.dev_threshold = dev_threshold
        self.transformer = transformer
        self.w_transform = w_transform
        self.columns = columns

        self.dfs_original = [ pd.DataFrame( data = [], index = [] ) for _ in range(nb_units) ]
        self.dfs = [ pd.DataFrame( data = [], index = [] ) for _ in range(nb_units) ] # FIXME: duplicates with self.detectors[i].df
        self.pg = PeerGrouping(self.w_ref_group)
        self.detectors = [ IndividualAnomalyInductive(w_martingale, non_conformity, k, dev_threshold) for _ in range(nb_units) ]
        self.transformers = [Transformer(w_transform, transformer) for _ in range(nb_units)]
        
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

        x_units_tr = [transformer.transform(x) for x, transformer in zip(x_units, self.transformers)]
        self.dfs = [append_to_df(self.dfs[i], dt, x) for i, x in enumerate(x_units_tr)]

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
    def get_similar_deviations(self, uid, from_time, to_time, k_devs=2, min_len=5, dev_threshold=None):
        target_devsig = self.detectors[uid].get_deviation_signature(from_time, to_time)
        deviations = [self.detectors[uuid].get_all_deviations(min_len, dev_threshold) for uuid in self.ids_target_units]

        deviations = []
        for uuid in self.ids_target_units:
            deviations_uuid = self.detectors[uuid].get_all_deviations(min_len, dev_threshold)
            deviations_uuid = [(devsig, p_from, p_to, uuid) for (devsig, p_from, p_to) in deviations_uuid]
            deviations += deviations_uuid

        dists = [np.linalg.norm(target_devsig - devsig) for (devsig, *_) in deviations]
        ids = np.argsort(dists)[:k_devs]
        return [deviations[id] for id in ids]

    # ===========================================
    def plot_deviations(self, figsize=None, savefig=None, plots=["data", "transformed_data", "strangeness", "deviation", "threshold"], debug=False):
        '''Plots the anomaly score, deviation level and p-value, over time.'''

        register_matplotlib_converters()

        if self.transformer is None and "transformed_data" in plots:
            plots.remove("transformed_data")

        plots, nb_axs, i = list(set(plots)), 0, 0
        if "data" in plots:
            nb_axs += 1
        if "transformed_data" in plots:
            nb_axs += 1
        if "strangeness" in plots:
            nb_axs += 1
        if any(s in ["pvalue", "deviation", "threshold"] for s in plots):
            nb_axs += 1

        fig, axs = plt.subplots(nb_axs, sharex="row", figsize=figsize)
        fig.autofmt_xdate()
        if not isinstance(axs, (np.ndarray) ): axs = np.array([axs])

        if "data" in plots:
            axs[i].set_xlabel("Time")
            axs[i].set_ylabel("Feature 0")
            for uid in self.ids_target_units:
                df = self.dfs_original[uid]
                axs[i].plot(df.index, df.values[:, 0], label="Unit {}".format(uid))
            axs[i].legend()
            i += 1

        if "transformed_data" in plots:
            axs[i].set_xlabel("Time")
            axs[i].set_ylabel("Trans. Feature 0")
            for uid in self.ids_target_units:
                df = self.dfs[uid]
                axs[i].plot(df.index, df.values[:, 0], label="Unit {}".format(uid))
                if debug and uid == self.ids_target_units[-1]:
                    T, representatives = self.detectors[uid].T, self.detectors[uid].representatives
                    axs[i].plot(T, np.array(representatives)[:, 0], label="Representative", ls="--", color="black")

            axs[i].legend()
            i += 1

        if "strangeness" in plots:
            axs[i].set_xlabel("Time")
            axs[i].set_ylabel("Strangeness")
            for uid in self.ids_target_units:
                T, S = self.detectors[uid].T, self.detectors[uid].S
                axs[i].plot(T, S, label="Unit {}".format(uid))
            axs[i].legend()
            i += 1

        if any(s in ["pvalue", "deviation", "threshold"] for s in plots):
            axs[i].set_xlabel("Time")
            axs[i].set_ylabel("Deviation")
            axs[i].set_ylim(0, 1)
            for uid in self.ids_target_units:
                T, P, M = self.detectors[uid].T, self.detectors[uid].P, self.detectors[uid].M
                if "pvalue" in plots:
                    axs[i].scatter(T, P, alpha=0.25, marker=".", color="green")
                if "deviation" in plots:
                    axs[i].plot(T, M)
                if "threshold" in plots:
                    axs[i].axhline(y=self.dev_threshold, color='r', linestyle='--')

        if savefig is None:
            plt.show()
        else:
            figpathname = utils.create_directory_from_path(savefig)
            plt.savefig(figpathname)

    # ===========================================
    def plot_explanations(self, uid, from_time, to_time, figsize=None, savefig=None, k_features=4):
        # TODO: validate if the period (from_time, to_time) has data before plotting
        detector = self.detectors[uid]
        sub_dfs_ori = [self.dfs_original[uuid][from_time: to_time] for uuid in range(self.nb_units)]
        sub_dfs = [self.dfs[uuid][from_time: to_time] for uuid in range(self.nb_units)]
        sub_representatives_df = pd.DataFrame(index=detector.T, data=detector.representatives)[from_time: to_time]
        sub_diffs_df = pd.DataFrame(index=detector.T, data=detector.diffs)[from_time: to_time]

        nb_features = sub_diffs_df.values.shape[1]
        if (self.columns is None) or (len(self.columns) != nb_features):
            self.columns = ["Feature {}".format(j) for j in range(nb_features)]
        self.columns = np.array(self.columns)

        features_scores = np.array([np.abs(col).mean() for col in sub_diffs_df.values.T])
        features_scores = 100 * features_scores / features_scores.sum()
        k_features = min(k_features, nb_features)
        selected_features_ids = np.argsort(features_scores)[-k_features:][::-1]
        selected_features_names = self.columns[selected_features_ids]
        selected_features_scores = features_scores[selected_features_ids]

        fig, axs = plt.subplots(k_features, 2, figsize=figsize)
        fig.autofmt_xdate()
        fig.suptitle("Ranked features of Unit {}\nFrom {} to {}".format(uid, from_time, to_time))
        if k_features == 1: axs = np.array([axs]).reshape(1, -1)

        for i, (j, name, score) in enumerate(zip(selected_features_ids, selected_features_names, selected_features_scores)):
            if i == 0: axs[i][0].set_title("Original data")
            axs[i][0].set_xlabel("Time")
            axs[i][0].set_ylabel("{0}\n(Score: {1:.1f})".format(name, score))
            for df_ori in sub_dfs_ori: axs[i][0].plot(df_ori.index, df_ori.values[:, j], color="silver")
            axs[i][0].plot(sub_dfs_ori[uid].index, sub_dfs_ori[uid].values[:, j], color="red")

            if i == 0: axs[i][1].set_title("Transformed data")
            axs[i][1].set_xlabel("Time")
            axs[i][1].set_ylabel("{}".format(name))
            for df in sub_dfs: axs[i][1].plot(df.index, df.values[:, j], color="silver")
            axs[i][1].plot(sub_representatives_df.index, sub_representatives_df.values[:, j], color="black", linestyle='--')
            axs[i][1].plot(sub_dfs[uid].index, sub_dfs[uid].values[:, j], color="red")

        figg = None
        if k_features > 1:
            figg, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
            figg.suptitle("Top 2 ranked features of Unit {}\nFrom {} to {}".format(uid, from_time, to_time))
            (j1, j2) , (nm1, nm2), (s1, s2) = selected_features_ids[:2], selected_features_names[:2], selected_features_scores[:2]

            ax1.set_title("Original data")
            ax1.set_xlabel("{0}\n(Score: {1:.1f})".format(nm1, s1))
            ax1.set_ylabel("{0}\n(Score: {1:.1f})".format(nm2, s2))
            for df_ori in sub_dfs_ori: ax1.scatter(df_ori.values[:, j1], df_ori.values[:, j2], color="silver", marker=".")
            ax1.scatter(sub_dfs_ori[uid].values[:, j1], sub_dfs_ori[uid].values[:, j2], color="red", marker=".", label="Unit {}".format(uid))
            ax1.legend()

            ax2.set_title("Transformed data")
            ax2.set_xlabel("{0}\n(Score: {1:.1f})".format(nm1, s1))
            ax2.set_ylabel("{0}\n(Score: {1:.1f})".format(nm2, s2))
            for df in sub_dfs: ax2.scatter(df.values[:, j1], df.values[:, j2], color="silver", marker=".")
            ax2.scatter(sub_dfs[uid].values[:, j1], sub_dfs[uid].values[:, j2], color="red", marker=".", label="Unit {}".format(uid))
            ax2.legend()

        if savefig is None:
            plt.show()
        else:
            figpathname = utils.create_directory_from_path(savefig)
            fig.savefig(figpathname)
            if figg is not None: figg.savefig(figpathname + "_2.png")
