import numpy as np
import pandas as pd
from days_delayed_lgbm.config import config
from sklearn.base import BaseEstimator, TransformerMixin


class DFMeanEncoding(BaseEstimator, TransformerMixin):
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode

    def fit(self,X,y=None):
        self.prior = 0
        self.alpha = 5
        self.mean_enc_settings = {}
        df = X.copy()
        self.target_col = 'target_col'
        df[self.target_col] = config.Y_TRAIN
        print("\n shape of df ",df.shape," \n")
        self.prior = df[self.target_col].mean() # global mean for Train fold
        if self.columns is not None:
            for col in self.columns: #iterate though the columns we want to encode
#                 if col != target_col:
                self.means     = df.groupby(col)[self.target_col].mean()
                self.nrows_cnt = df.groupby(col)[self.target_col].count()
                self.means_reg = (self.means*self.nrows_cnt + self.prior*self.alpha) / (self.nrows_cnt + self.alpha)
                self.mean_enc_settings.update({
                     col: self.means_reg.to_dict()

                })
        return self

    def transform(self,X,y=None):
        '''
        Transforms columns of X specified in self.columns using
         If no columns specified, transforms all
        columns in X.
        '''
        df = X.copy()
        for col, means in self.mean_enc_settings.items(): #iterate though the columns we want to encode
            df[col] = df[col].map(means).astype(np.float32).fillna(self.prior)
        return df

    #def fit_transform(self,X,y=None):
    #    return self.fit(X,y).transform(X)
