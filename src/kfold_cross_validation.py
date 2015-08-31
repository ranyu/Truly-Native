import xgboost as xgb
import cPickle as pickle

import pandas as pd
import numpy as np

from scipy.sparse import csr_matrix, hstack
from sklearn.decomposition import TruncatedSVD
from sklearn import metrics
from sklearn import cross_validation
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
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
    if model == "xgb":
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test)
        bst = xgb.train(param, dtrain, num_round)
        prediction = bst.predict(dtest)

    elif model in ["rf", "lr", "lsvm"]:
        if model == "rf":
            clf = RandomForestClassifier(n_estimators=500, random_state=1234, max_features=100, n_jobs=4)
        elif model == "lr":
            clf = LogisticRegression(C=3.0, random_state=1234)
        else:
            clf = LinearSVC(C=0.5, random_state=1234)
        clf.fit(X_train, y_train)
        prediction = clf.predict_proba(X_test)
        prediction = prediction[:, 1]

    else:
        raise ValueError("model must be Random Forest(rf), GBDT(xgb), \
                         Logistic Regression(lr) or Linear SVM(lsvm)")
        
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


def main(is_apply_svd, model):
    
    with open("datasets/trainTagSparse.pkl") as f:
        X_tag = pickle.load(f)
    with open("datasets/trainAttrSparse.pkl") as f:
        X_attr = pickle.load(f)
    with open("datasets/y.pkl") as f:
        label = pickle.load(f)
        y = np.array([int(i) for i in label])
        
    y = y[:NUM_OF_SAMPLE]
    X = csr_matrix(hstack((X_tag, X_attr)))
    X = X[:NUM_OF_SAMPLE]

    if is_apply_svd:
        print "Performing SVD..."
        svd = TruncatedSVD(n_components=150, n_iter=5)
        X = svd.fit_transform(X)

    print "Training..."
    score = k_fold_validation(X, y, 5, model)
    print "Average Error: " + str(score)


if __name__ == "__main__":
    main(is_apply_svd=False, model="lsvm")
