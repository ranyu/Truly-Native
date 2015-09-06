import xgboost as xgb
import cPickle as pickle

import pandas as pd
import numpy as np

from scipy.sparse import csr_matrix, hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


with open("datasets/train_idx.pkl") as f:
    train_idx = pickle.load(f)
with open("datasets/test_idx.pkl") as f:
    test_idx = pickle.load(f)
with open("datasets/y.pkl") as f:
    label = pickle.load(f)
    y = np.array([int(i) for i in label])


def training(model):
    print "reading training data..."
    with open("datasets/trainTagSparse.pkl") as f:
        X_tag = pickle.load(f)
    with open("datasets/trainAttrSparse.pkl") as f:
        X_attr = pickle.load(f)
    with open("datasets/trainValueSparse.pkl") as f:
        X_value = pickle.load(f)
    X = csr_matrix(hstack((X_tag, X_attr, X_value)))
    X_tag = X_attr = X_value = None

    print "reading test data..."
    with open("datasets/testTagSparse.pkl") as f:
        X_test_tag = pickle.load(f)
    with open("datasets/testAttrSparse.pkl") as f:
        X_test_attr = pickle.load(f)
    with open("datasets/testValueSparse.pkl") as f:
        X_test_value = pickle.load(f)
    X_test = csr_matrix(hstack((X_test_tag, X_test_attr, X_test_value)))
    X_test_tag = X_test_attr = X_test_value = None

    if model != "lr":
        print "Performing SVD..."
        svd = TruncatedSVD(n_components=200, n_iter=5)
        X = svd.fit_transform(X)
        X_test = svd.transform(X_test)

    if model == "xgb":
        print "Using Xgboost..."
        dtrain = xgb.DMatrix(X, label=y)
        param = {"objective":"binary:logistic", "nthread":8,
                 "eval_metric":"auc", "bst:max_depth":30, 
                 "bst:min_child_weight":1, "bst:subsample":0.7,
                 "bst:colsample_bytree":0.7, "bst:eta":0.01}
        num_round = 1200
        print "Training..."
        bst = xgb.train(param, dtrain, num_round)
        print "Predicting..."
        dtest = xgb.DMatrix(X_test)
        prediction = bst.predict(dtest)

    elif model == "lr":
        print "Using Logistic Regression..."
        clf = LogisticRegression(C=3.0, random_state=1234)
        clf.fit(X, y)
        prediction = clf.predict_proba(X_test)
        prediction = prediction[:, 1]

    elif model == "rf":
        print "Using Random Forest..."
        forest = RandomForestClassifier(n_estimators=500, random_state=1234, max_features=100, n_jobs=4)
        forest.fit(X, y)
        prediction = forest.predict_proba(X_test)
        prediction = prediction[:, 1]

    else:
        raise ValueError("model must be Random Forest(rf), GBDT(xgb) or Logistic Regression(lr)")

    return prediction


if __name__ == "__main__":
    prediction = training(model="xgb")
    submission = pd.DataFrame({"id": test_idx, "prediction": prediction})
    submission.to_csv("submissions/xgboost.csv", index=False)
