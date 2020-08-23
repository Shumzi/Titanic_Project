from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import scale
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import xgboost as xgb
from catboost import CatBoostClassifier, Pool, cv
import lightgbm

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score

from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from time import time


class Preprocessor():
    """
    A preprocessor transformer class for the Titanic dataset.
    The preprocessing is as follows:
    - Use initials of names as a feature.
    - Fill NaN ages by the mean age grouped by Initials.
    - parse deck level from 'Cabin' feature.
    - drop Fare, Name, Cabin, Ticket as they were used for another feature/aren't worth the trouble.
    - one-hot categorical data.
    """

    def fit_transform(self, data):
        """
        fit the preprocessor parameters to the training data,
        and transform the train data as well
        :param data:
        X_train data, to fit the data preprocessor transformer.
        :return:
        The one_hot_encoder and age groups needed to recreate preprocessing for cv/test set
        (use transform func on cv/test set).
        The other feature creation can be recreated without reference to the train set.
        """
        data = data.copy()
        data.reset_index(inplace=True, drop=True)
        # name_to_initials
        data['Initial'] = data['Name'].apply(lambda x: x.split(',')[1].split()[0])
        freq_initials = data['Initial'].value_counts()
        freq_initials = freq_initials[freq_initials > 4].index
        data.loc[~data['Initial'].isin(freq_initials), 'Initial'] = 'Rare'
        # fill age NaNs by initials
        ages = data.groupby('Initial')['Age'].mean()
        data.Age = data.apply(lambda x: ages[x['Initial']] if np.isnan(x['Age']) else x['Age'], axis=1)
        # cabin_to_deck_no (keep NaNs as is)
        splitted_cabin = data['Cabin'].dropna().str.split(pat='(\d+)').apply(lambda x: x[0])
        data['Cabin_Deck'] = pd.DataFrame(splitted_cabin.tolist(), index=splitted_cabin.index)
        # as sex has only 2 categories, just binarize it.
        data['Sex'] = (data['Sex'] == 'male').astype(int)
        # ticket: just shows family relation
        # name: we already extracted initials
        # fare: is just a convoluted way of saying pclass.
        # cabin: we used it for deck feature, no need for it now.
        data.drop(self.drop_columns, axis=1, inplace=True)
        # one_hot categories all at once, drop originals afterwards.
        ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
        ohe_array = ohe.fit_transform(data[self.to_enc_array].fillna(-1).astype(str))
        ohe_df = pd.DataFrame(ohe_array, columns=ohe.get_feature_names())
        data = pd.concat([data, ohe_df], axis='columns').drop(self.to_enc_array, axis='columns')
        age_scaler = StandardScaler()
        data['Age'] = age_scaler.fit_transform(data['Age'].values.reshape(-1, 1))
        return data, ages, freq_initials, ohe, age_scaler

    def transform(self, data):
        """
        transform the test/cv data to the same format as the train data.
        :param data:
        X_test/X_cv test set.
        :return:
        preprocessed data.
        """
        data = data.copy()
        data.reset_index(inplace=True, drop=True)
        # name_to_initials
        data['Initial'] = data['Name'].apply(lambda x: x.split(',')[1].split()[0])
        print(data['Initial'].head())
        data.loc[~data['Initial'].isin(self.freq_initials), 'Initial'] = 'Rare'
        print(data['Initial'].value_counts())
        # fill age NaNs by initials
        print(self.ages.index)
        X.Age = X.apply(lambda x: self.ages[x['Initial']] if np.isnan(x['Age']) else x['Age'], axis=1)
        # cabin_to_deck_no (keep NaNs as is)
        splitted_cabin = data['Cabin'].dropna().str.split(pat='(\d+)').apply(lambda x: x[0])
        data['Cabin_Deck'] = pd.DataFrame(splitted_cabin.tolist(), index=splitted_cabin.index)
        # as sex has only 2 categories, just binarize it.
        data['Sex'] = (data['Sex'] == 'male').astype(int)
        data.drop(self.drop_columns, axis=1, inplace=True)
        # one_hot categories all at once, drop originals afterwards.
        ohe_array = self.ohe.transform(data[self.to_enc_array].fillna(-1).astype(str))
        ohe_df = pd.DataFrame(ohe_array, columns=self.ohe.get_feature_names())
        data = pd.concat([data, ohe_df], axis='columns').drop(self.to_enc_array, axis='columns')
        data['Age'] = self.age_scaler.transform(data['Age'].values.reshape(-1, 1))
        return data

    def __init__(self, X_train):
        self.drop_columns = ['Ticket', 'Name', 'Fare', 'Cabin']
        self.to_enc_array = ['Pclass', 'Initial', 'SibSp', 'Parch', 'Embarked', 'Cabin_Deck']
        self.train_data, self.ages, self.freq_initials, self.ohe, self.age_scaler = self.fit_transform(X_train)