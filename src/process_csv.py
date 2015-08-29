import csv
import sys
import cPickle as pickle
csv.field_size_limit(999999999)

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.feature_extraction.text import TfidfVectorizer


PATH_TO_TRAIN = "datasets/trainTags.csv"
PATH_TO_TEST = "datasets/testTags.csv"


def get_feature(path, col):
    """ Get feature values for given column index
    parameters:
    --------------------------------------------------------
    path: path to the file
    col: column index
    """
    csvfile = csv.reader(open(path))
    feature = [i[col] for t, i in enumerate(csvfile) if t!= 0]
    return feature


def get_train_matrix(path):
    """ Transform training data into sparse matrices
    parameters:
    --------------------------------------------------------
    path: path to the training file
    """
    idx = get_feature(path, 0)
    tags = get_feature(path, 1)
    attrs = get_feature(path, 2)
    y = get_feature(path, 3)
    
    tfv_tag = TfidfVectorizer(min_df=1, max_features=None, strip_accents='unicode',
                              analyzer='word', token_pattern=r'\w{1,}',
                              ngram_range=(1, 3), use_idf=1, smooth_idf=1,
                              sublinear_tf=1, stop_words=None)
    tfv_attr = TfidfVectorizer(min_df=1, max_features=None, strip_accents='unicode',
                               analyzer='word', token_pattern=r'\w{1,}',
                               ngram_range=(1, 1), use_idf=1, smooth_idf=1,
                               sublinear_tf=1, stop_words=None)
    tags = tfv_tag.fit_transform(tags)
    attrs = tfv_attr.fit_transform(attrs)

    return idx, tags, attrs, y, tfv_tag, tfv_attr


def get_test_matrix(path, tfv_tag, tfv_attr):
    """ Transform test data into sparse matrices
    parameters:
    --------------------------------------------------------
    path: path to the test file
    tfv_tag: the trained tfidf vectorizer for tag
    tfv_attr: the trained tfidf vectorizer for attribute
    """
    idx = get_feature(path, 0)
    tags = get_feature(path, 1)
    attrs = get_feature(path, 2)

    tags = tfv_tag.transform(tags)
    attrs = tfv_attr.transform(attrs)

    return idx, tags, attrs


print "get train matrices"
train_idx, train_tags, train_attrs, y, tfv_tag, tfv_attr = get_train_matrix(PATH_TO_TRAIN)
           
print "save train matrices"
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
train_tags = train_attrs = None

print "get test matrices"
test_idx, test_tags, test_attrs = get_test_matrix(PATH_TO_TEST, tfv_tag, tfv_attr)

print "save test matrices"
with open("datasets/test_idx.pkl", "w") as f:
    pickle.dump(test_idx, f)
    f.close()
with open("datasets/testTagSparse.pkl", "w") as f:
    pickle.dump(test_tags, f)
    f.close()
with open("datasets/testAttrSparse.pkl", "w") as f:
    pickle.dump(test_attrs, f)
    f.close()
