import pandas as pd, sqlite3
import numpy as np
from datetime import datetime
from os import listdir
from os.path import isfile, join

class Data:
    '''Example class to generates data from several units.
    '''
    def __init__(self, data_path, freq="1d", start=datetime(2012, 1, 1), end=datetime(2012, 3, 10)):
        self.ts_init = ( datetime(year=2011, month=6, day=1) - datetime(1970, 1, 1) ).total_seconds() * 1000
        
        self.dbfiles = [ join(data_path, f) for f in listdir(data_path) if isfile(join(data_path, f)) ]
        self.dates = pd.date_range(start=start, end=end, freq=freq).to_pydatetime()
        
        self.generators = [ self.windowing(dbfile) for dbfile in self.dbfiles ]
    
    # -------------------------------------
    def date2ts(self, date):
        return ( date - datetime(1970, 1, 1) ).total_seconds() * 1000 - self.ts_init
        
    # -------------------------------------
    def read_data_chunk(self, dbfile, t_from, t_to):
        t_from, t_to = self.date2ts(t_from), self.date2ts(t_to)
        conn = sqlite3.connect(dbfile)
        query = "SELECT timestamp, value FROM data WHERE "+str(t_from)+" < timestamp AND timestamp < "+str(t_to)
        ts = pd.read_sql_query(query, conn, index_col="timestamp", parse_dates={'timestamp': {'format': 'ms'}}).ix[:,0]
        if len(ts) < 3: return None
        ts.index = ts.index + self.ts_init
        ts.index = pd.to_datetime(ts.index, unit='ms')
        conn.close()
        return ts
        
    # -------------------------------------
    def windowing(self, dbfile):
        X, T = [], []
        for i in range(len(self.dates)-1):
            ts = self.read_data_chunk(dbfile, self.dates[i], self.dates[i+1])
            yield pd.DataFrame(data=[], index=[]) if ts is None else pd.DataFrame(data = ts.values, index = ts.index)
            
    # -------------------------------------
    def getNext(self):
        for time in self.dates:
            dfs = [next(generator) for generator in self.generators]
            yield time, dfs
        
