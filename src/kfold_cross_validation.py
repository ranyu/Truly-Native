import xgboost as xgb
import cPickle as pickle

import pandas as pd
import numpy as np

from scipy.sparse import csr_matrix, hstack
from sklearn.decomposition import TruncatedSVD
from sklearn import metrics
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier


NUM_OF_SAMPLE = 10000
param = {"objective":"binary:logistic", "nthread":8,
         "eval_metric":"auc", "bst:max_depth":30, 
         "bst:min_child_weight":1, "bst:subsample":0.7,
         "bst:colsample_bytree":0.7, "bst:eta":0.01}
num_round = 1200


def train_and_test(X_train, X_test, y_train, y_test, model):
    """Build model in training set, and test the performance in test set
    Parameters
    --------------------------------------------------------------------
    train (numpy.array): training data in numpy array format
    test (numpy.array): test data in numpy array format
    y_train (numpy.array): target of training data
    y_test (numpy.array): target of test data
    model (str): name of the classifier
    Outputs
    --------------------------------------------------------------------
    loss (float): error of the model in test data
    """
    if model == "rf":
        forest = RandomForestClassifier(n_estimators=100, random_state=1234)
        forest = forest.fit(X_train, y_train)
        prediction = forest.predict_proba(X_test)
        prediction = prediction[:, 1]
        
    elif model == "xgb":
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test)
        bst = xgb.train(param, dtrain, num_round)
        prediction = bst.predict(dtest)
        
    y_test = np.array(y_test)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, prediction, pos_label=1)
    loss = metrics.auc(fpr, tpr)
    print "Error of current folder: " + str(loss)
    return loss


def k_fold_validation(data, y, trials=5, model="rf"):
    """Use cross validation to evaluate the performance of model
    Parameters
    --------------------------------------------------------------------
    data (numpy.array): values of independent variables
    y (numpy.array): values of dependent variable
    trials (int): number of folders
    model (str): name of the classifier
    Outputs
    --------------------------------------------------------------------
    loss (float): average error of the model from cross validation
    """
    skf = cross_validation.StratifiedKFold(y, n_folds=trials)
    error = 0.0
    
    for train_index, test_index in skf:
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = y[train_index], y[test_index]
        error += train_and_test(X_train, X_test, y_train, y_test, model)
        
    return error/trials


with open("datasets/trainTagSparse.pkl") as f:
    X_tag = pickle.load(f)

with open("datasets/trainAttrSparse.pkl") as f:
    X_attr = pickle.load(f)

with open("datasets/y.pkl") as f:
    label = pickle.load(f)
    y = np.array([int(i) for i in label])

with open("datasets/train_idx.pkl") as f:
    train_idx = pickle.load(f)

with open("datasets/test_idx.pkl") as f:
    test_idx = pickle.load(f)

print "Performing SVD..."
X = hstack((X_tag, X_attr))
svd = TruncatedSVD(n_components=200, n_iter=5)
X = svd.fit_transform(X)
X = X[:NUM_OF_SAMPLE]
y = y[:NUM_OF_SAMPLE]

print "Training..."
score = k_fold_validation(X, y, 5, "xgb")
print score
