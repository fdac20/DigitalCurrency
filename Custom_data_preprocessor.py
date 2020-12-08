import csv
import numpy as np
from Data_preprocessor import DataGenerator
import pandas as pd
from stockstats import StockDataFrame as Sdf
from sklearn import preprocessing
#class to calcute the indexes and preprocess data
class data_processor(DataGenerator):
    @staticmethod
    def _generator(filename, header=False, split=0.8, mode='train',spread=.005):
        df = pd.read_csv(filename)
        if "Name" in df:
            df.drop('Name',axis=1,inplace=True)
        _stock = Sdf.retype(df.copy())
        _stock.get('cci_14')
        _stock.get('rsi_14')
        _stock.get('dx_14')
        _stock = _stock.dropna(how='any')
        min_max_scaler = preprocessing.MinMaxScaler((-1, 1))
        np_scaled = min_max_scaler.fit_transform(_stock[['rsi_14', 'cci_14','dx_14','volume']])
        df_normalized = pd.DataFrame(np_scaled)
        df_normalized.columns = ['rsi_14', 'cci_14','dx_14','volume']
        df_normalized['bid'] = _stock['close'].values
        df_normalized['ask'] = df_normalized['bid'] + spread
        df_normalized['mid'] = (df_normalized['bid'] + df_normalized['ask'])/2
        split_len=int(split*len(df_normalized))
        if(mode=='train'):
            raw_data = df_normalized[['ask','bid','mid','rsi_14','cci_14','dx_14','volume']].iloc[:split_len,:]
        else:
            raw_data = df_normalized[['ask', 'bid', 'mid', 'rsi_14', 'cci_14','dx_14','volume']].iloc[split_len:,:]

        for index, row in raw_data.iterrows():
            yield row.as_matrix()
#Rewinds if end of data reached.
    def _iterator_end(self):
        super(self.__class__, self).rewind()
#For this generator, we want to rewind only when the end of the data is reached.
    def rewind(self):
        self._iterator_end()
