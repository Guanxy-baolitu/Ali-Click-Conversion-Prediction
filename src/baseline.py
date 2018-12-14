#-*-coding:utf-8-*-
"""
将特征直接喂给xgboost or lr
"""
import re
import os
from sklearn.datasets.svmlight_format import load_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn.metrics.ranking import roc_auc_score
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import numpy as np

from preprocessing import csv_to_libsvm

def load_data(data_path):
    X_all, y_all = load_svmlight_file(data_path)
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size = 0.3, random_state = 42)
    return X_train, X_test, y_train, y_test

def train(X_train, y_train, method = 'lr'):
    if method == 'xgb':
        clf = xgb.XGBClassifier(scale_pos_weight=187, 
                            learning_rate=0.08,
                            n_estimators=500,
                            max_depth=10,
                            gamma=0,
                            subsample=0.9,
                            colsample_bytree=0.5,
                            objective='binary:logistic'
                        )
    elif method == 'lr':
        clf = LogisticRegression()

    clf.fit(X_train, y_train)
    return clf

def save_model(model, model_path):
    model.save_model(model_path)

def load_model(model_path):
    return xgb.Booster(model_file=model_path)

def evaluate(model, X_test, y_test):
    y_pred_test_lr = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_test_lr)
    print('xgb test auc: %.5f' % auc)

if __name__ == '__main__':
    feature_path = ''
    libsvm_path = os.path.join('data', 'model_test.txt')
    model_path = ''
    # csv_to_libsvm(feature_path, libsvm_path)
    X_train, X_test, y_train, y_test = load_data(libsvm_path)
    model = train(X_train, y_train, method='lr')
    # save_model(model, model_path)
    # model = load_model(model_path)
    evaluate(model, X_test, y_test)


