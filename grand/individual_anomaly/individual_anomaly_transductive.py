"""A transductive anomaly detection model for an individual system.
To detect anomalies, the IndividualAnomalyTransductive model compares the data from
a system against its own past historical data from the stream.
"""

__author__ = "Mohamed-Rafik Bouguelia"
__license__ = "MIT"
__email__ = "mohamed-rafik.bouguelia@hh.se"

from grand.conformal import pvalue, Strangeness
from grand.utils import DeviationContext, InputValidationError, append_to_df, dt2num
from grand import utils
import matplotlib.pylab as plt, pandas as pd, numpy as np
from pandas.plotting import register_matplotlib_converters


class IndividualAnomalyTransductive:
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

    def __init__(self, w_martingale=15, non_conformity="median", k=20, dev_threshold=0.6, ref_group=["season-of-year"], external_percentage=0.3):
        utils.validate_individual_deviation_params(w_martingale, non_conformity, k, dev_threshold, ref_group)

        self.w_martingale = w_martingale
        self.non_conformity = non_conformity
        self.k = k
        self.dev_threshold = dev_threshold
        self.ref_group = ref_group
        self.external_percentage = external_percentage

        self.strg = Strangeness(non_conformity, k)
        self.scores = []
        self.T, self.S, self.P, self.M = [], [], [], []

        self.mart = 0
        self.marts = [0, 0, 0]

        self.df = pd.DataFrame(index=[], data=[])
        self.externals = []

        self.df_init = pd.DataFrame(index=[], data=[])
        self.externals_init = []

    # ===========================================
    def predict(self, dtime, x, external=None):
        '''Update the deviation level based on the new test sample x

        Parameters:
        -----------
        dtime : datetime
            datetime corresponding to the sample x

        x : array-like, shape (n_features,)
            Sample for which the strangeness, p-value and deviation level are computed

        external: float (default None)
            Used in case self.ref_group == "external" to construct the reference dataset from historical data

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
        self._fit(dtime, x, external)

        strangeness = self.strg.get(x)
        self.S.append(strangeness)

        pval = pvalue(strangeness, self.scores)
        self.P.append(pval)

        deviation = self._update_martingale(pval)
        self.M.append(deviation)

        is_deviating = deviation > self.dev_threshold
        return DeviationContext(strangeness, pval, deviation, is_deviating)

    # ===========================================
    def init(self, data):
        # TODO: check if data is a list of elements i with (t_i, x_i, [external_i])
        # TODO: and that if ref_group="external" then external_i should be specified (i.e. len(data[0]) == 3)
        if len(data[0]) == 2:
            times, X = list(zip(*data))
            externals = []

        elif len(data[0]) == 3:
            times, X, externals = list(zip(*data))

        self.df_init = pd.DataFrame(index=times, data=X)
        self.externals_init = externals

        return self

    # ===========================================
    def _fit(self, dtime, x, external=None):
        ''' Private method for internal use only.
        Constructs a reference dataset based on historical data and the specified ref_group criteria
        and fits a model to this reference data.
        '''

        if self.ref_group == "external":
            if external is None:
                raise InputValidationError("When ref_group is set to 'external', the parameter external must be specified.")

            all_externals = np.array( list(self.externals_init) + list(self.externals) )
            all_X = np.array( list(self.df_init.values) + list(self.df.values) )

            k = int( len(all_externals) * self.external_percentage )
            ids = np.argsort( np.abs(all_externals - external) )[:k]
            X = all_X[ids]
        else:
            df_sub = self.df.append(self.df_init)
            for criterion in self.ref_group:
                current = dt2num(dtime, criterion)
                historical = np.array([dt2num(dt, criterion) for dt in df_sub.index])
                df_sub = df_sub.loc[(current == historical)]
            X = df_sub.values

        if len(X) == 0:
            X = [x]

        self.strg.fit(X)
        self.scores = self.strg.get_fit_scores()

        self.df = append_to_df(self.df, dtime, x)
        self.externals.append(external)

    # ===========================================
    def _update_martingale(self, pval):
        ''' Private method for internal use only.
        Incremental additive martingale over the last w_martingale steps.

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

        normalized_mart = (mat_in_window) / (.5 * w)
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
