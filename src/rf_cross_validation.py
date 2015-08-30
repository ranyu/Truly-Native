import cPickle as pickle

import pandas as pd
import numpy as np

from scipy.sparse import csr_matrix, hstack
from sklearn.decomposition import TruncatedSVD
from sklearn import metrics
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier


NUM_OF_SAMPLE = 10000


def train_and_test(X_train, X_test, y_train, y_test):
    forest = RandomForestClassifier(n_estimators=100, random_state=1234)
    forest = forest.fit(X_train, y_train)
    proba = forest.predict_proba(X_test)
    proba = proba[:, 1]
    y_test = np.array(y_test)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, proba, pos_label=1)
    loss = metrics.auc(fpr, tpr)
    print "Error of current folder: " + str(loss)
    return loss


def k_fold_validation(data, y, trials=10):
    skf = cross_validation.StratifiedKFold(y, n_folds=10)
    error = 0.0
    for train_index, test_index in skf:
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = y[train_index], y[test_index]
        error += train_and_test(X_train, X_test, y_train, y_test)
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
score = k_fold_validation(X, y, 5)
print score
