"""Provides ways to transform the data of each system, so that the data from
various systems is more fairly comparable.
"""

__author__ = "Mohamed-Rafik Bouguelia"
__license__ = "MIT"
__email__ = "mohamed-rafik.bouguelia@hh.se"

import numpy as np
from scipy.stats import linregress
from grand import utils


class Transformer:
    def __init__(self, w=20, transformer="pvalue"):
        self.w = w
        self.transformer = transformer
        self.X = None
        self.P = None
        self.MIN_HISTORY_SIZE = 10

        # TODO: validate the parameters (accepted strings) for self.transformer

    def transform_all(self, X):
        Z = np.array([self.transform(x) for x in X])
        return Z

    def transform(self, x):
        if len(x) == 0 or self.transformer is None:
            return x

        if self.X is None and self.P is None:
            self.X = np.empty((0, len(x)))
            self.P = np.empty((0, len(x)))

        if self.transformer in ["mean_normalize", "std_normalize", "mean_std_normalize"]:
            with_mean = (self.transformer != "std_normalize")
            with_std = (self.transformer != "mean_normalize")
            return self.transform_normalize(x, with_mean, with_std)

        elif self.transformer in ["pvalue", "mean_pvalue"]:
            aggregate = (self.transformer == "mean_pvalue")
            return self.transform_pvalue(x, aggregate)

        elif self.transformer in ["slope"]:
            return self.transform_slope(x)

        else:
            raise utils.InputValidationError("transformer should be one of {}".format(["mean_normalize", "std_normalize", "mean_std_normalize", "slope", "pvalue", "mean_pvalue"]))


    def transform_slope(self, x):
        self.X = np.vstack([self.X, x])
        if len(self.X) < self.w: return np.zeros(x.shape)
        return np.array([ linregress(range(self.w), self.X[-self.w:, j])[0] for j in range(len(x)) ])


    def transform_normalize(self, x, with_mean=True, with_std=True):
        self.X = np.vstack([self.X, x])
        x_normalized = np.array(x)

        if with_mean:
            x_normalized -= np.mean(self.X, axis=0)

        if with_std:
            std = np.std(self.X, axis=0)
            x_normalized /= (std if std>0 else 1)

        return x_normalized

    def transform_pvalue(self, x, aggregate=True):
        if len(self.X) < self.MIN_HISTORY_SIZE:
            p_arr = [0.5 for _ in range(len(x))]
        else:
            pvalue = lambda val, values: len([1 for v in values if v > val]) / len(values)
            p_arr = [pvalue(x[i], self.X[:, i]) for i in range(len(x))]

        self.P = np.vstack([self.P, p_arr])
        self.X = np.vstack([self.X, x])

        z = []
        for i in range(len(x)):
            pvalues = self.P[-self.w:, i]
            if aggregate:
                z += [1. - np.mean(pvalues)] if len(pvalues) >= 1 else [0.5]
            else:
                z += [1. - pvalues[-1]] if len(pvalues) >= 1 else [0.5]

        return z
