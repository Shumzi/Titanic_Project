from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import numpy as np

def cross_val_score_regular(est,X,y,get_errors = False):
    '''
    built in cross_val_score func overestimates results for some reason,
    so this is a simple implementation of cross_val_score for roc_auc.
    :param est: estimator to test generalizability upon
    :param X: data
    :param y: labels
    :return: cv scores, mean of cv scores, train scores, mean of train scores.
    '''
    kf = StratifiedKFold()
    scores = np.zeros(5)
    scores_train = np.zeros(5)
    for i, (train_index, test_index) in enumerate(kf.split(X, y)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_cv = y.iloc[train_index], y.iloc[test_index]
        est.fit(X_train,y_train)
        cv_pred = est.predict(X_test)
        scores[i] = roc_auc_score(y_cv, cv_pred)
        scores_train[i] = roc_auc_score(y_train, est.predict(X_train))
        if get_errors:
            print(confusion_matrix(y_cv, cv_pred), '\n')
            if i == 0:
                print('fp: \n', X_test[(y_cv != cv_pred) & (y_cv == 0)])
                fn = X_test[(y_cv != cv_pred) & (y_cv == 1)]
                print('\n\nfn: \n', fn)
    return scores, scores.mean(), scores_train, scores_train.mean()
def get_errors_and_features(pipe, X, y):
    '''
    display the errors a given pipeline makes.
    assumes that the pipe contains an estimator with a 'feature importances' field,
    :param pipe:
    :param X:
    :param y:
    :return: feature importances of the first fold, including the feature names (size: 2 x num_of_features),
    false positives and false negatives.
    '''
    kf = StratifiedKFold()
    train_scores = np.zeros(5)
    cv_scores = np.zeros(5)
    for i, (train_index, cv_index) in enumerate(kf.split(X, y)):
        Xtrain = X.iloc[train_index].reset_index(drop=True)
        ytrain = y.iloc[train_index].reset_index(drop=True)
        Xcv = X.iloc[cv_index].reset_index(drop=True)
        ycv = y.iloc[cv_index].reset_index(drop=True)
        pipe.fit(Xtrain, ytrain)
        pred_cv = pipe.predict(Xcv)
        pred_train = pipe.predict(Xtrain)
        conf = confusion_matrix(ycv, pred_cv)
        print(conf)
        if i == 0:
            fp = Xcv[(ycv != pred_cv) & (ycv == 0)]
            print('fp: \n', fp)
            fn = Xcv[(ycv != pred_cv) & (ycv == 1)]
            print('\n\nfn: \n', fn)
            feature_importances = np.c_[pipe['classifier'].feature_importances_,
                                        pipe['pre_processing'].fit_transform(Xtrain, ytrain).columns]
        train_scores[i] = (roc_auc_score(ytrain, pred_train))
        cv_scores[i] = (roc_auc_score(ycv, pred_cv))
        print('train score: ', roc_auc_score(ytrain, pred_train))
        print('cv score: ', roc_auc_score(ycv, pred_cv),'\n')
    print('\n\ntrain scores: ', train_scores, '\nmean: ', train_scores.mean())
    print('cv scores: ', cv_scores, '\nmean: ', cv_scores.mean())
    return feature_importances,fp,fn

class OrdinalEncoderWithUnknown():
    '''
    Basic ordinal encoder, with the ability to handle unknowns
    by putting them into a new ordinal category.
    '''

    def __init__(self, cat_index='all'):
        self.dicts = {}
        # cat_index is the categorical feature index list
        self.cat_index = cat_index

    def fit(self, df, *y):
        if self.cat_index == 'all':
            self.cat_index = list(range(df.shape[1]))
        for feat in self.cat_index:
            dic = np.unique(df.iloc[:, feat])
            dic = dict([(unique, index) for index, unique in enumerate(dic)])
            self.dicts[feat] = dic

    def transform(self, df):
        df_output = df.copy()
        for feat in self.cat_index:
            dic = self.dicts[feat]
            df_output.iloc[:, feat] = df.iloc[:, feat].apply(self.unknown_value, args=(dic,))
        return df_output

    def fit_transform(self, df, *y):
        self.fit(df, y)
        return self.transform(df, y)
        # if self.cat_index=='all':
        #     self.cat_index=list(range(df.shape[1]))
        # df_output=df.copy()
        # for feat in self.cat_index:
        #     dic=np.unique(df.iloc[:,feat])
        #     dic=dict([(unique,index) for index, unique in enumerate(dic)])
        #     self.dicts[feat]=dic
        #     df_output.iloc[:,feat]=df.iloc[:,feat].apply(lambda x: dic[x])
        # return df_output

    def unknown_value(self, value, dic):
        '''
        create separate value of unknown inputs.
        '''
        try:
            return dic[value]
        except:
            return len(dic)