import csv
import sys
import cPickle as pickle
csv.field_size_limit(sys.maxsize)

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.feature_extraction.text import TfidfVectorizer


PATH_TO_TRAIN = "datasets/trainTags.csv"
PATH_TO_TEST = "datasets/testTags.csv"


def get_train_feature(path_to_train, col):
    traincsv = csv.reader(open(path_to_train))
    feature = []
    for t, i in enumerate(traincsv):
        if t == 0:
            continue
        feature.append(i[col])
    return feature


def get_test_matrix(path_to_test, tfv, col, n):
    """ Transform one column of test data into sparse matrix
    parameters:
    --------------------------------------------------------
    path_to_test: path to the test file
    tfv: the trained tfidf vectorizer
    col: the column to be transformed
    n: width of the output matrix
    """
    testcsv = csv.reader(open(path_to_test))
    idx = [i[0] for t, i in enumerate(testcsv) if t!= 0]
    m = len(idx)
    testcsv = csv.reader(open(path_to_test))
    sparse_matrix = lil_matrix((m, n))

    for t, line in enumerate(testcsv):
        if t == 0:
            continue
        sparse_matrix[t-1] = tfv.transform([line[col]])

    return idx, csr_matrix(sparse_matrix)


print "get tags matrix for train"
train_idx = get_train_feature(PATH_TO_TRAIN, 0)
y = get_train_feature(PATH_TO_TRAIN, 3)
train_tags = get_train_feature(PATH_TO_TRAIN, 1)
tfv_tag = TfidfVectorizer(min_df=1, max_features=None, strip_accents='unicode',
                          analyzer='word', token_pattern=r'\w{1,}',
                          ngram_range=(1, 2), use_idf=1, smooth_idf=1,
                          sublinear_tf=1, stop_words=None)
train_tags = tfv_tag.fit_transform(train_tags)

print "get attributes matrix for train"
train_attrs = get_train_feature(PATH_TO_TRAIN, 2)
tfv_attr = TfidfVectorizer(min_df=1, max_features=None, strip_accents='unicode',
                           analyzer='word', token_pattern=r'\w{1,}',
                           ngram_range=(1, 1), use_idf=1, smooth_idf=1,
                           sublinear_tf=1, stop_words=None)
train_attrs = tfv_attr.fit_transform(train_attrs)

print "save the matrices"
with open("datasets/y.pkl", "w") as f:
    pickle.dump(y, f)
    f.close()
with open("datasets/train_idx.pkl", "w") as f:
    pickle.dump(train_idx, f)
    f.close()
with open("datasets/trainTagSparse.pkl", "w") as f:
    pickle.dump(train_tags, f)
    f.close()
with open("datasets/trainAttrSparse.pkl", "w") as f:
    pickle.dump(train_attrs, f)
    f.close()

# collect garbage
m, n1 = train_tags.shape
m, n2 = train_attrs.shape
train_tags = train_attrs = None

print "get tags matrix for test"
test_idx, test_tags = get_test_matrix(PATH_TO_TEST, tfv_tag, 1, n1)
with open("datasets/test_idx.pkl", "w") as f:
    pickle.dump(test_idx, f)
    f.close()
with open("datasets/testTagSparse.pkl", "w") as f:
    pickle.dump(test_tags, f)
    f.close()

# collect garbage
test_tags = None

print "get atrributes matrix for test"
test_idx, test_attrs = get_test_matrix(PATH_TO_TEST, tfv_attr, 2, n2)
with open("datasets/testAttrSparse.pkl", "w") as f:
    pickle.dump(test_attrs, f)
    f.close()
