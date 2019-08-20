import numpy as np
from grand.conformal import pvalue


class Transformer:
    def __init__(self, w=20):
        self.w = w
        self.X = None
        self.P = None
        self.MIN_HISTORY_SIZE = 10

    def aggregate(self, i):
        pvalues = self.P[-self.w:, i]
        return [1. - np.mean(pvalues)] if len(pvalues) >= 1 else [0.5]

    def transform(self, x):
        if len(x) == 0:
            return x

        if self.X is None and self.P is None:
            self.X = np.empty((0, len(x)))
            self.P = np.empty((0, len(x)))

        if len(self.X) < self.MIN_HISTORY_SIZE:
            p_arr = [0.5 for i in range(len(x))]
        else:
            p_arr = [pvalue(x[i], self.X[:, i]) for i in range(len(x))]

        self.P = np.vstack([self.P, p_arr])
        self.X = np.vstack([self.X, x])

        z = []
        for i in range(len(x)):
            z += self.aggregate(i)

        return z

    def transform_all(self, X):
        Z = np.array([self.transform(x) for x in X])
        return Z
