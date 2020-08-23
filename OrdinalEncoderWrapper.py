import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.base import BaseEstimator,TransformerMixin
class OrdinalEncoderWithUnkown(BaseEstimator,TransformerMixin):
    """
    Ordinal Encoder with option for unknown input.
    """
    def __init__(self):
        self.ord_enc = OrdinalEncoder()

    def fit_transform(self,X):
        # save unique vals of each col in case of unknown in transform stage
        # and add unknown
        self.unique = dict((c,X[c].unique()) for c in X.columns)
        X = X.append(pd.DataFrame(np.full((1,X.shape[1]),'Unknown')))
        print('Ord enc\n',X)
        print(X.isna().sum())
        return self.ord_enc.fit_transform(X)

    def transform(self,X):
        X = X.copy()
        for c in X.columns:
            X[c] = X[c].apply(lambda x: 'Unknown' if x not in self.unique[c] else x)
        return self.ord_enc.transform(X)