"""An inductive anomaly detection model for an individual system.
To detect anomalies, the IndividualAnomalyInductive model compares the data from
a system against a fixed reference dataset.
"""

__author__ = "Mohamed-Rafik Bouguelia"
__license__ = "MIT"
__email__ = "mohamed-rafik.bouguelia@hh.se"

from grand.conformal import pvalue, Strangeness
from grand.utils import DeviationContext, append_to_df
from grand import utils
import matplotlib.pylab as plt, pandas as pd, numpy as np
from pandas.plotting import register_matplotlib_converters


class IndividualAnomalyInductive:
    '''Deviation detection for a single/individual unit
    
    Parameters:
    ----------
    w_martingale : int
        Window used to compute the deviation level based on the last w_martingale samples. 
                
    non_conformity : string
        Strangeness (or non-conformity) measure used to compute the deviation level.
        It must be either "median" or "knn"
                
    k : int
        Parameter used for k-nearest neighbours, when non_conformity is set to "knn"
        
    dev_threshold : float
        Threshold in [0,1] on the deviation level
    '''
    
    def __init__(self, w_martingale=15, non_conformity="median", k=20, dev_threshold=0.6):
        utils.validate_individual_deviation_params(w_martingale, non_conformity, k, dev_threshold)
        
        self.w_martingale = w_martingale
        self.non_conformity = non_conformity
        self.k = k
        self.dev_threshold = dev_threshold
        
        self.strg = Strangeness(non_conformity, k)
        self.scores = []
        self.T, self.S, self.P, self.M = [], [], [], []
        
        self.mart = 0
        self.marts = [0, 0, 0]

        self.df = pd.DataFrame(data=[], index=[])
        
    # ===========================================
    def fit(self, X):
        '''Fit the anomaly detector to the data X (assumed to be normal)
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples assumed to be not deviating from normal
        
        Returns:
        --------
        self : object
        '''
        
        self.strg.fit(X)
        self.scores = self.strg.get_fit_scores()
        
        return self
    
    # ===========================================
    def predict(self, dtime, x):
        '''Update the deviation level based on the new test sample x
        
        Parameters:
        -----------
        dtime : datetime
            datetime corresponding to the sample x
        
        x : array-like, shape (n_features,)
            Sample for which the strangeness, p-value and deviation level are computed
        
        Returns:
        --------
        strangeness : float
            Strangeness of x with respect to samples in Xref
        
        pval : float, in [0, 1]
            p-value that represents the proportion of samples in Xref that are stranger than x.
        
        deviation : float, in [0, 1]
            Normalized deviation level updated based on the last w_martingale steps
        '''
        
        self.T.append(dtime)
        self.df = append_to_df(self.df, dtime, x)
        
        strangeness = self.strg.get(x)
        self.S.append(strangeness)
        
        pval = pvalue(strangeness, self.scores)
        self.P.append(pval)
        
        deviation = self._update_martingale(pval)
        self.M.append(deviation)
        
        is_deviating = deviation > self.dev_threshold
        return DeviationContext(strangeness, pval, deviation, is_deviating)
        
    # ===========================================
    def _update_martingale(self, pval):
        '''Incremental additive martingale over the last w_martingale steps.
        
        Parameters:
        -----------
        pval : int, in [0, 1]
            The most recent p-value
        
        Returns:
        --------
        normalized_one_sided_mart : float, in [0, 1]
            Deviation level. A normalized version of the current martingale value.
        '''
        
        betting = lambda p: -p + .5
        
        self.mart += betting(pval)
        self.marts.append(self.mart)
        
        w = min(self.w_martingale, len(self.marts))
        mat_in_window = self.mart - self.marts[-w]
        
        normalized_mart = ( mat_in_window ) / (.5 * w)
        normalized_one_sided_mart = max(normalized_mart, 0)
        return normalized_one_sided_mart
        
    # ===========================================
    def get_stats(self):
        stats = np.array([self.S, self.M, self.P]).T
        return pd.DataFrame(index=self.T, data=stats, columns=["strangeness", "deviation", "pvalue"])

    # ===========================================
    def plot_deviations(self, figsize=None, savefig=None, plots=["data", "strangeness", "pvalue", "deviation", "threshold"]):
        '''Plots the anomaly score, deviation level and p-value, over time.
        '''

        register_matplotlib_converters()

        plots, nb_axs, i = list(set(plots)), 0, 0
        if "data" in plots:
            nb_axs += 1
        if "strangeness" in plots:
            nb_axs += 1
        if any(s in ["pvalue", "deviation", "threshold"] for s in plots):
            nb_axs += 1

        print("nb_axs", nb_axs)

        fig, axes = plt.subplots(nb_axs, sharex="row", figsize=figsize)
        if not isinstance(axes, (np.ndarray) ):
            axes = np.array([axes])

        if "data" in plots:
            axes[i].set_title("Data")
            axes[i].set_xlabel("Time")
            axes[i].set_ylabel("Feature 0")
            axes[i].plot(self.df.index, self.df.values[:, 0])
            i += 1

        if "strangeness" in plots:
            axes[i].set_xlabel("Time")
            axes[i].set_ylabel("Strangeness")
            axes[i].plot(self.T, self.S)
            i += 1

        if any(s in ["pvalue", "deviation", "threshold"] for s in plots):
            axes[i].set_xlabel("Time")
            axes[i].set_ylabel("Deviation")

            if "pvalue" in plots:
                axes[i].scatter(self.T, self.P, alpha=0.25, marker=".", color="green", label="p-value")

            if "deviation" in plots:
                axes[i].plot(self.T, self.M, label="Deviation")

            if "threshold" in plots:
                axes[i].axhline(y=self.dev_threshold, color='r', linestyle='--', label="Threshold")

            axes[i].legend()

        fig.autofmt_xdate()

        if savefig is None:
            plt.draw()
            plt.show()
        else:
            figpathname = utils.create_directory_from_path(savefig)
            plt.savefig(figpathname)
