import pandas as pd
import glob
import os
from scipy.io import loadmat
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import numpy as np
import time
from timeit import default_timer as timer
no_of_estimators=2000
datapath='./fedata40/'



def compute_accuracy(clf, train_x, train_y, test_x, test_y):
    clfResult=clf.fit(train_x, train_y)
    clf_pred = clf.predict(test_x)
    acc_score = metrics.accuracy_score(test_y, clf_pred)
    return acc_score


def loadData(filenameX, filenameY, filenameX_test, filenameY_test):

    "Loading " + filenameX + "..."
    dataX = loadmat(filenameX, squeeze_me=True)
    X = dataX['Training_x']

    "Loading " + filenameY + "..."
    dataY = loadmat(filenameY, squeeze_me=True)
    Y = dataY['Training_y']

    "Loading " + filenameX_test + "..."
    dataX_test = loadmat(filenameX_test, squeeze_me=True)
    X_test = dataX_test['Testing_x']

    "Loading " + filenameY_test + "..."
    dataY_test = loadmat(filenameY_test, squeeze_me=True)
    Y_test = dataY_test['Testing_y']

    X = X - np.mean(X, axis=0)
    X = X / np.std(X, axis=0)

    X_test = X_test - np.mean(X_test, axis=0)
    X_test = X_test / np.std(X_test, axis=0)
    return X, Y, X_test, Y_test

filenameX_train = os.path.join(datapath, "X_train.mat")
filenamey_train = os.path.join(datapath, "y_train.mat")
filenameX_test = os.path.join(datapath, "X_test.mat")
filenamey_test = os.path.join(datapath, "y_test.mat")

X_train, y_train, X_test, y_test = loadData(filenameX=filenameX_train, filenameY=filenamey_train ,
                                    filenameX_test=filenameX_test, filenameY_test=filenamey_test)

extra_clf = ExtraTreesClassifier(n_estimators=no_of_estimators, n_jobs=1)
extra_clf_acc = compute_accuracy(extra_clf, X_train, y_train, X_test, y_test)
print('Extra Trees Accuracy : %.6f' % extra_clf_acc)
print("2000 estimators")