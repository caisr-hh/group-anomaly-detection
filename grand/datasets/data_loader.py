"""Provides an easy way to load data from several CSV files in a repository and stream it.
Each CSV file contains the dataset of one unit. The class provides an easy way to stream the data from
all units.
"""

__author__ = "Mohamed-Rafik Bouguelia"
__license__ = "MIT"
__email__ = "mohamed-rafik.bouguelia@hh.se"

import pandas as pd, numpy as np, matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
from os import listdir
from os.path import isfile, join, dirname


# ===================================================================================
class DataCSVs:
    '''Helper class to read data from several units.
    The data of each unit is presented as a CSV file.
    '''
    def __init__(self, data_path):
        self.files_csv = sorted([ join(data_path, f) for f in listdir(data_path) if isfile(join(data_path, f)) ])
        self.dfs, self.dates = [], []
    
    # -------------------------------------
    def load(self):
        for file_csv in self.files_csv:
            df = pd.read_csv(file_csv, index_col=0, parse_dates=[0]).dropna()
            self.dfs.append(df)
            
        index = self.dfs[0].index
        self.dates = index.union_many([ df.index for df in self.dfs ]).to_pydatetime()
        return self

    # -------------------------------------
    def get_nb_features(self):
        return self.dfs[0].values.shape[1]

    # -------------------------------------
    def get_nb_units(self):
        return len(self.files_csv)
        
    # -------------------------------------
    def normalize(self, with_mean=True, with_std=True):
        if with_mean:
            self.dfs = [df - df.mean() for df in self.dfs]

        if with_std:
            stds = [df.std() for df in self.dfs]
            for std in stds: std[std==0] = 1
            self.dfs = [df / std for df, std in zip(self.dfs, stds)]

        return self

    # -------------------------------------
    def stream(self):
        for dt in self.dates:
            x_units = []
            for df in self.dfs:
                try: x_units.append( df.loc[dt].values )
                except: x_units.append( np.array([]) )
                
            yield dt, x_units
            
    # -------------------------------------
    def stream_unit(self, i):
        for dt in self.dates:
            df = self.dfs[i]
            try: yield dt, df.loc[dt].values
            except: pass

    # -------------------------------------
    def plot(self, icol=0, max_units=6, figsize=None):
        register_matplotlib_converters()
        fig, ax = plt.subplots(figsize = figsize)
        ax.set_xlabel("Time")
        ax.set_ylabel("Feature {}".format(icol))

        for i, df in enumerate(self.dfs):
            if i == max_units: break
            ax.plot(df.index, df.values[:, icol], label="Unit {}".format(i))

        plt.legend()
        fig.autofmt_xdate()
        plt.show()


# ===================================================================================
def loader(foldname):
    data_path = join(dirname(__file__), 'data', foldname)
    return DataCSVs(data_path).load()


# ===================================================================================
def load_vehicles():
    return loader('vehicles')


def load_artificial(i, smoothing=1):
    data_loader = loader("toy" + str(int(i%8)))
    data_loader.dfs = [df.rolling(window=smoothing).mean().dropna() for df in data_loader.dfs]
    data_loader.dfs[2][:] *= 2  # double the values from Unit 2
    return data_loader


def load_taxi():
    return loader("taxi")

