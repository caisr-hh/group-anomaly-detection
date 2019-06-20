from grand.conformal import pvalue, Strangeness
from grand.utils import DeviationContext, InputValidationError, append_to_df
from grand import utils

import matplotlib.pylab as plt, pandas as pd, numpy as np


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

    def __init__(self, w_martingale=15, non_conformity="median", k=20, dev_threshold=0.6, ref_group="month"):
        utils.validate_individual_deviation_params(w_martingale, non_conformity, k, dev_threshold, ref_group)

        self.w_martingale = w_martingale
        self.non_conformity = non_conformity
        self.k = k
        self.dev_threshold = dev_threshold
        self.ref_group = ref_group

        self.strg = Strangeness(non_conformity, k)
        self.scores = []
        self.T, self.S, self.P, self.M = [], [], [], []

        self.mart = 0
        self.marts = [0, 0, 0]

        self.df = pd.DataFrame( data = [], index = [] )
        self.externals = []

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
    def _fit(self, dtime, x, external=None):
        ''' Private method for internal use only.
        Constructs a reference dataset based on historical data and the specified ref_group criteria
        and fits a model to this reference data.
        '''

        if self.ref_group == "week":
            current = dtime.isocalendar()[1]
            historical = np.array([dt.isocalendar()[1] for dt in self.df.index])
            X = self.df.loc[(current == historical)].values

        elif self.ref_group == "month":
            current = dtime.month
            historical = np.array([dt.month for dt in self.df.index])
            X = self.df.loc[(current == historical)].values

        elif self.ref_group == "season":
            season = {12:1, 1:1, 2:1, 3:2, 4:2, 5:2, 6:3, 7:3, 8:3, 9:4, 10:4, 11:4}
            get_season = lambda dt: season[dt.month]
            current = get_season(dtime)
            historical = np.array([get_season(dt) for dt in self.df.index])
            X = self.df.loc[(current == historical)].values

        else: # self.ref_group == "external":
            if external is None:
                raise InputValidationError("When ref_group is set to 'external', the parameter external must specified.")

            current = external
            historical = np.array(self.externals)

            pm = 2 * np.std(historical) / 10 if len(historical) > 0 else 0
            X = self.df.loc[(current-pm <= historical) & (historical <= current+pm)].values

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
    def plot_deviations(self):
        '''Plots the anomaly score, deviation level and p-value, over time.
        '''

        fig = plt.figure(0)
        plt.title("Anomaly scores over time")
        plt.xlabel("Time")
        plt.ylabel("Anomaly score")
        plt.plot(self.T, self.S)
        fig.autofmt_xdate()

        fig = plt.figure(1)
        plt.title("Deviation level and p-values over time")
        plt.xlabel("Time")
        plt.ylabel("Deviation level")
        plt.scatter(self.T, self.P, alpha=0.25, marker=".", color="green", label="p-value")
        plt.plot(self.T, self.M, label="deviation")
        plt.axhline(y=self.dev_threshold, color='r', linestyle='--', label="Threshold")
        plt.legend()
        fig.autofmt_xdate()

        fig = plt.figure(2)
        plt.title("Data")
        plt.xlabel("Time")
        plt.ylabel("Feature value")
        plt.plot(self.df.index, self.df.values[:, 0], marker=".", label="Feature 0")
        if self.df.values.shape[1] > 1:
            plt.plot(self.df.index, self.df.values[:, 1], marker=".", label="Feature 1")
        plt.legend()
        fig.autofmt_xdate()

        plt.show()
