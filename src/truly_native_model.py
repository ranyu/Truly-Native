import xgboost as xgb
import cPickle as pickle

import pandas as pd
import numpy as np

from scipy.sparse import csr_matrix, hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

##############################################################################
# parameters #################################################################
##############################################################################

IS_APPLY_SVD = False
MODEL = "lr"


with open("datasets/train_idx.pkl") as f:
    train_idx = pickle.load(f)
with open("datasets/test_idx.pkl") as f:
    test_idx = pickle.load(f)
with open("datasets/y.pkl") as f:
    label = pickle.load(f)
    y = np.array([int(i) for i in label])


def training(model):
    """ Apply given model to fit the training data
    Parameters
    ---------------------------------------------------
    model: name of the model
    Output
    ---------------------------------------------------
    svd: the SVD transformer if applying dimensionality
         reduction
    clf: the classifier that fit the data
    """
    print "reading training data..."
    with open("datasets/trainTagSparse.pkl") as f:
        X_tag = pickle.load(f)
    with open("datasets/trainAttrSparse.pkl") as f:
        X_attr = pickle.load(f)
    with open("datasets/trainTitleSparse.pkl") as f:
        X_title = pickle.load(f)
    with open("datasets/trainTextSparse.pkl") as f:
        X_text = pickle.load(f)

    if IS_APPLY_SVD:
        with open("datasets/trainValueSparse2.pkl") as f:
            X_value = pickle.load(f)
    else:
        with open("datasets/trainValueSparse.pkl") as f:
            X_value = pickle.load(f)

    X = csr_matrix(hstack((X_tag, X_attr, X_value, X_title, X_text)))
    X_tag = X_attr = X_value = X_title = X_text = None

    if IS_APPLY_SVD:
        print "applying SVD..."
        svd = TruncatedSVD(n_components=200, n_iter=5)
        X = svd.fit_transform(X)

    if model == "xgb":
        print "applying xgboost..."
        dtrain = xgb.DMatrix(X, label=y)
        param = {"objective":"binary:logistic", "nthread":8,
                 "eval_metric":"auc", "bst:max_depth":30, 
                 "bst:min_child_weight":1, "bst:subsample":0.7,
                 "bst:colsample_bytree":0.7, "bst:eta":0.01}
        num_round = 1200
        print "training..."
        clf = xgb.train(param, dtrain, num_round)
    elif model == "lr":
        print "Using Logistic Regression..."
        clf = LogisticRegression(C=3.0, random_state=1234)
        clf.fit(X, y)
    elif model == "rf":
        print "Using Random Forest..."
        clf = RandomForestClassifier(n_estimators=500, random_state=1234, max_features=100, n_jobs=4)
        clf.fit(X, y)
    else:
        raise ValueError("model must be Random Forest(rf), GBDT(xgb) or Logistic Regression(lr)")

    if IS_APPLY_SVD:
        return svd, clf
    return None, clf


def predicting(svd, clf, model):
    """ Use the trained model to make prediction
    Parameters
    ---------------------------------------------------
    svd: the SVD transformer if applying dimensionality
         reduction
    clf: the classifier that fit the data
    Output
    ---------------------------------------------------
    prediction: the prediction made by classifier
    """
    print "reading test data..."
    with open("datasets/testTagSparse.pkl") as f:
        X_test_tag = pickle.load(f)
    with open("datasets/testAttrSparse.pkl") as f:
        X_test_attr = pickle.load(f)
    with open("datasets/testTitleSparse.pkl") as f:
        X_test_title = pickle.load(f)
    with open("datasets/testTextSparse.pkl") as f:
        X_test_text = pickle.load(f)

    if IS_APPLY_SVD:
        with open("datasets/testValueSparse2.pkl") as f:
            X_test_value = pickle.load(f)
    else:
        with open("datasets/testValueSparse.pkl") as f:
            X_test_value = pickle.load(f)

    X_test = csr_matrix(hstack((X_test_tag, X_test_attr, X_test_value,
                                X_test_title, X_test_text)))
    X_test_tag = X_test_attr = X_test_value = X_test_title = X_test_text = None

    if IS_APPLY_SVD:
        print "applying SVD..."
        X_test = svd.transform(X_test)

    print "predicting..."
    if model == "xgb":
        dtest = xgb.DMatrix(X_test)
        prediction = clf.predict(dtest)
    elif model in ["lr", "rf"]:
        prediction = clf.predict_proba(X_test)
        prediction = prediction[:, 1]
    else:
        raise ValueError("model must be Random Forest(rf), GBDT(xgb) or Logistic Regression(lr)")

    return prediction


if __name__ == "__main__":
    svd, clf = training(MODEL)
    prediction = predicting(svd, clf, MODEL)
    submission = pd.DataFrame({"id": test_idx, "prediction": prediction})
    submission.to_csv("submissions/logistic.csv", index=False)
