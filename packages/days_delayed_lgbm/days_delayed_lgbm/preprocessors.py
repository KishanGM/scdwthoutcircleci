import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from days_delayed_lgbm import config


class DFMeanEncoding(BaseEstimator, TransformerMixin):
    def __init__(self,columns = None,df_target_col = None):
        self.columns = columns # array of column names to encode
        self.df_target_col = df_target_col

    def fit(self,X,y=None):
        mean_enc_settings = {}
        df = X.copy()
        #target_col = 'late_delivery'
        df[target_col] = self.df_target_col
        prior_loc = df[target_col].mean() # global mean for Train fold
        global prior
        prior = prior_loc
        if self.columns is not None:
            for col in self.columns: #iterate though the columns we want to encode
#                 if col != target_col:
                means     = df.groupby(col)[target_col].mean()
                nrows_cnt = df.groupby(col)[target_col].count()
                means_reg = (means*nrows_cnt + config.prior*config.alpha) / (nrows_cnt + config.alpha)
                mean_enc_settings.update({
                     col: means_reg.to_dict()

                })
                config.mean_enc_settings = mean_enc_settings
        return self

    def transform(self,X,y=None):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        df = X.copy()
        for col, means in mean_enc_settings.items(): #iterate though the columns we want to encode
            df[col] = df[col].map(means).astype(np.float32).fillna(prior)
        return df

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)
