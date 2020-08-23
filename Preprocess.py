
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.metrics import f1_score, plot_roc_curve, auc
from sklearn.metrics import roc_auc_score

from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from time import time

from sklearn.base import BaseEstimator,TransformerMixin
import scipy
from sklearn.utils.fixes import loguniform
from scipy.stats import uniform
import textwrap
import matplotlib.pyplot as plt

# Do not use from sklearn.preprocessing import _BaseEncoder, it is protected class!
from sklearn.preprocessing._encoders import _BaseEncoder
class new_OrdinalEncoder(_BaseEncoder):
    def __init__(self,cat_index='all'):
        self.dicts={}
        # cate_index is the categorical feature index list
        self.cat_index=cat_index

    def fit(self,df,*y):
        if self.cat_index=='all':
            self.cat_index=list(range(df.shape[1]))
        for feat in self.cat_index:
            dic=np.unique(df.iloc[:,feat])
            dic=dict([(i,index) for index, i in enumerate(dic)])
            self.dicts[feat]=dic

    def fit_transform(self,df,*y):
        if self.cat_index=='all':
            self.cat_index=list(range(df.shape[1]))
        df_output=df.copy()
        for feat in self.cat_index:
            dic=np.unique(df.iloc[:,feat])
            dic=dict([(i,index) for index, i in enumerate(dic)])
            self.dicts[feat]=dic
            df_output.iloc[:,feat]=df.iloc[:,feat].apply(lambda x: dic[x])
        return df_output

    def transform(self,df):
        df_output=df.copy()
        for feat in self.cat_index:
            dic=self.dicts[feat]
            df_output.iloc[:,feat]=df.iloc[:,feat].apply(self.unknown_value,args=(dic,))
        return df_output

    def unknown_value(self,value,dic): # It will set up a new interger for unknown values!
        try:
            return dic[value]
        except:
            return len(dic)


class Preprocessor(BaseEstimator,TransformerMixin):
    """
    A preprocessor transformer class for the Titanic dataset.
    The preprocessing stages are as follows:\n
    - Use title of names as a feature.\n
    - Fill NaN ages by the mean age grouped by Title.\n
    - parse deck level from 'Cabin' feature.\n
    - drop Name, Cabin, Ticket as they were used for another feature/aren't worth the trouble.\n
    - one-hot categorical data.
    """

    def fit_transform(self, X, *y):
        """
        fit the preprocessor parameters to the training data,
        and transform the train data as well
        :param data:
        X_train data, to fit the data preprocessor transformer.
        :return:
        -------
        X_new : ndarray array of shape (n_samples, n_features_new)
            Transformed array.
        """
        X = X.copy()
        X.reset_index(inplace=True,drop=True)
        # name_to_initials
        X['Initial'] = X['Name'].apply(lambda x: x.split(',')[1].split()[0])
        self.freq_initials = X['Initial'].value_counts()
        self.freq_initials = self.freq_initials[self.freq_initials>4].index
        X.loc[~X['Initial'].isin(self.freq_initials),'Initial'] = 'Rare'
        # fill age NaNs by initials
        self.ages = X.groupby('Initial')['Age'].mean()
        X.Age = X.apply(lambda x : self.ages[x['Initial']] if np.isnan(x['Age']) else x['Age'],axis=1)
        # cabin_to_deck_no (keep NaNs as is)
        splitted_cabin = X['Cabin'].dropna().str.split(pat='(\d+)').apply(lambda x: x[0])
        X['Cabin_Deck'] = pd.DataFrame(splitted_cabin.tolist(), index = splitted_cabin.index)
        # as sex has only 2 categories, just binarize it.
        X['Sex'] = (X['Sex']=='male').astype(int)
        # ticket: just shows family relations, which is covered in parch and sibsp anyway.
        # name: we already extracted initials
        # fare: is just a convoluted way of saying pclass.
        # cabin: we used it for deck feature, no need for it now.
        X.drop(self.drop_columns,axis=1,inplace=True)
        # one_hot categories all at once, drop originals afterwards.
        if(self.one_hot_cat):
            self.ohe = OneHotEncoder(handle_unknown='ignore',sparse=False)
            ohe_array = self.ohe.fit_transform(X[self.cat_columns].fillna(-1).astype(str))
            ohe_df = pd.DataFrame(ohe_array, columns=self.ohe.get_feature_names())
            X = pd.concat([X,ohe_df],axis='columns').drop(self.cat_columns,axis='columns')
        else:
            # catboost/LightGBM
            X[self.cat_text_cols] = X[self.cat_text_cols].fillna('nan')
            self.ord_enc = new_OrdinalEncoder()
            X[self.cat_text_cols] = self.ord_enc.fit_transform(X[self.cat_text_cols])
            # X[self.cat_text_cols+self.cat_columns] = X[self.cat_text_cols+self.cat_columns].astype('category')
        self.age_scaler = RobustScaler()
        X['Age'] = self.age_scaler.fit_transform(X['Age'].values.reshape(-1,1))
        return X

    def fit(self,X, *y):
        dummy = self.fit_transform(X, *y)

    def transform(self,X, *y):
        """
        transform the test/cv data to the same format as the train data.
        :param X:
        X_test/X_cv test set.
        :return:
        preprocessed X.
        """
        X = X.copy()
        X.reset_index(inplace=True,drop=True)
        # name_to_Title
        X['Initial'] = X['Name'].apply(lambda x: x.split(',')[1].split()[0])
        X.loc[~X['Initial'].isin(self.freq_initials),'Initial'] = 'Rare'
        # fill age NaNs by initials
        X.Age = X.apply(lambda x : self.ages[x['Initial']] if np.isnan(x['Age']) else x['Age'],axis=1)
        # cabin_to_deck_no (keep NaNs as is)
        splitted_cabin = X['Cabin'].dropna().str.split(pat='(\d+)').apply(lambda x: x[0])
        X['Cabin_Deck'] = pd.DataFrame(splitted_cabin.tolist(), index = splitted_cabin.index)
        # as sex has only 2 categories, just binarize it.
        X['Sex'] = (X['Sex'] == 'male').astype(int)
        X.drop(self.drop_columns,axis=1,inplace=True)
        # one_hot categories all at once, drop originals afterwards.
        if(self.one_hot_cat):
            ohe_array = self.ohe.transform(X[self.cat_columns].fillna(-1).astype(str))
            ohe_df = pd.DataFrame(ohe_array, columns=self.ohe.get_feature_names())
            X = pd.concat([X,ohe_df],axis='columns').drop(self.cat_columns,axis='columns')
        else:
            X[self.cat_text_cols] = self.ord_enc.transform(X[self.cat_text_cols].fillna('nan')).astype(int)
            # X[self.cat_text_cols+self.cat_columns] = X[self.cat_text_cols+self.cat_columns].astype('category')
        X['Age'] = self.age_scaler.transform(X['Age'].values.reshape(-1,1))
        return X


    def __init__(self,one_hot_cat = True):
        self.one_hot_cat = one_hot_cat
        self.drop_columns = ['Ticket','Name','Cabin']
        self.cat_columns = ['Pclass','Initial','SibSp','Parch','Embarked','Cabin_Deck']
        self.cat_text_cols = ['Initial','Embarked','Cabin_Deck']

# Do not use from sklearn.preprocessing import _BaseEncoder, it is protected class!
from sklearn.preprocessing._encoders import _BaseEncoder
class new_OrdinalEncoder(_BaseEncoder):
    def __init__(self,cat_index='all'):
        self.dicts={}
        # cate_index is the categorical feature index list
        self.cat_index=cat_index

    def fit(self,df,*y):
        if self.cat_index=='all':
            self.cat_index=list(range(df.shape[1]))
        for feat in self.cat_index:
            dic=np.unique(df.iloc[:,feat])
            dic=dict([(i,index) for index, i in enumerate(dic)])
            self.dicts[feat]=dic

    def fit_transform(self,df,*y):
        if self.cat_index=='all':
            self.cat_index=list(range(df.shape[1]))
        df_output=df.copy()
        for feat in self.cat_index:
            dic=np.unique(df.iloc[:,feat])
            dic=dict([(i,index) for index, i in enumerate(dic)])
            self.dicts[feat]=dic
            df_output.iloc[:,feat]=df.iloc[:,feat].apply(lambda x: dic[x])
        return df_output

    def transform(self,df):
        df_output=df.copy()
        for feat in self.cat_index:
            dic=self.dicts[feat]
            df_output.iloc[:,feat]=df.iloc[:,feat].apply(self.unknown_value,args=(dic,))
        return df_output

    def unknown_value(self,value,dic): # It will set up a new interger for unknown values!
        try:
            return dic[value]
        except:
            return len(dic)
