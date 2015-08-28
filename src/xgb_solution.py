import xgboost as xgb
import cPickle as pickle

import pandas as pd
import numpy as np

from scipy.sparse import csr_matrix, hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler


with open("datasets/trainTagSparse.pkl") as f:
    X_tag = pickle.load(f)

with open("datasets/trainAttrSparse.pkl") as f:
    X_attr = pickle.load(f)

##with open("datasets/testTagSparse.pkl") as f:
##    X_test_tag = pickle.load(f)
##
##with open("datasets/testAttrSparse.pkl") as f:
##    X_test_attr = pickle.load(f)

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

print "Using Xgboost..."
dtrain = xgb.DMatrix(X, label=y)
param = {"objective":"binary:logistic", "nthread":8,
         "eval_metric":"auc", "bst:max_depth":30, 
         "bst:min_child_weight":1, "bst:subsample":1.0,
         "bst:colsample_bytree":1.0, "bst:eta":0.05}
num_round = 300

print "Training..."
bst = xgb.cv(param, dtrain, num_round, nfold=5)
##bst = xgb.train(param, dtrain, num_round)
##
##print "Predicting..."
##dtest = xgb.DMatrix(X_test)
##prediction = bst.predict(dtest)
##scalar = MinMaxScaler()
##prediction = scalar.fit_transform(prediction)
##submission = pd.DataFrame({"id": test_idx, "prediction": prediction})
##submission.to_csv("submissions/xgboost.csv", index=False)
