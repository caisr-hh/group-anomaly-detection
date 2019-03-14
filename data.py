import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join

class Data:
    '''Helper class to read data from several units.
    The data of each unit is presented as a CSV file.
    '''
    def __init__(self, data_path):
        self.files_csv = [ join(data_path, f) for f in listdir(data_path) if isfile(join(data_path, f)) ]
        self.dfs, self.dates = [], []
    
    # -------------------------------------
    def load(self):
        for file_csv in self.files_csv:
            df = pd.read_csv(file_csv, index_col=0, parse_dates=[0]).dropna()
            self.dfs.append(df)
            
        index = self.dfs[0].index
        self.dates = index.union_many([ df.index for df in self.dfs ])#.to_pydatetime()
        return self
        
    # -------------------------------------
    def getNext(self):
        for dt in self.dates:
            x_units = []
            for df in self.dfs:
                try: x_units.append( df.loc[dt].values )
                except: x_units.append( np.array([]) )
                
            yield dt, x_units
            
            