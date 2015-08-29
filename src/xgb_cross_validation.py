import xgboost as xgb
import cPickle as pickle

import pandas as pd
import numpy as np

from scipy.sparse import csr_matrix, hstack
from sklearn.decomposition import TruncatedSVD


NUM_OF_SAMPLE = 10000

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

print "Using Xgboost..."
dtrain = xgb.DMatrix(X, label=y)
param = {"objective":"binary:logistic", "nthread":8,
         "eval_metric":"auc", "bst:max_depth":30, 
         "bst:min_child_weight":1, "bst:subsample":0.7,
         "bst:colsample_bytree":0.7, "bst:eta":0.01}
num_round = 1000

print "Training..."
bst = xgb.cv(param, dtrain, num_round, nfold=5)
