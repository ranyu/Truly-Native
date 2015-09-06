import csv
import sys
import cPickle as pickle
csv.field_size_limit(999999999)

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.feature_extraction.text import TfidfVectorizer


PATH_TO_TRAIN = "datasets/trainTags.csv"
PATH_TO_TEST = "datasets/testTags.csv"


def get_train_matrix(path):
    """ Transform training data into sparse matrices
    parameters:
    --------------------------------------------------------
    path: path to the training file
    """
    csvfile = csv.reader(open(path))
    idx = []
    tags = []
    attrs = []
    values = []
    y = []
    
    for t, row in enumerate(csvfile):
        # skip title
        if t == 0:
            continue
        if t % 10000 == 0:
            print "encountered: %d" % t 
        idx.append(row[0])
        tags.append(row[1])
        attrs.append(row[2])
        values.append(row[3])
        y.append(row[4])

    print "applying tfidf to training data..."
    tfv_tag = TfidfVectorizer(min_df=1, max_features=None, strip_accents='unicode',
                              analyzer='word', token_pattern=r'\w{1,}',
                              ngram_range=(1, 3), use_idf=1, smooth_idf=1,
                              sublinear_tf=1, stop_words=None)
    tfv_attr = TfidfVectorizer(min_df=1, max_features=None, strip_accents='unicode',
                               analyzer='word', token_pattern=r'\w{1,}',
                               ngram_range=(1, 2), use_idf=1, smooth_idf=1,
                               sublinear_tf=1, stop_words=None)
    tfv_val = TfidfVectorizer(min_df=1, max_features=None, strip_accents='unicode',
                              analyzer='word', token_pattern=r'\w{1,}',
                              ngram_range=(1, 2), use_idf=1, smooth_idf=1,
                              sublinear_tf=1, stop_words=None)
    
    tags = tfv_tag.fit_transform(tags)
    attrs = tfv_attr.fit_transform(attrs)
    values = tfv_val.fit_transform(values)

    return idx, tags, attrs, values, y, tfv_tag, tfv_attr, tfv_val


def get_test_matrix(path, tfv_tag, tfv_attr, tfv_val):
    """ Transform test data into sparse matrices
    parameters:
    --------------------------------------------------------
    path: path to the test file
    tfv_tag: the trained tfidf vectorizer for tag
    tfv_attr: the trained tfidf vectorizer for attribute
    tfv_val: the trained tfidf vectorizer for value
    """
    csvfile = csv.reader(open(path))
    idx = []
    tags = []
    attrs = []
    values = []
    
    for t, row in enumerate(csvfile):
        # skip title
        if t == 0:
            continue
        if t % 10000 == 0:
            print "encountered: %d" % t 
        idx.append(row[0])
        tags.append(row[1])
        attrs.append(row[2])
        values.append(row[3])

    print "applying tfidf to test data..."
    tags = tfv_tag.transform(tags)
    attrs = tfv_attr.transform(attrs)
    values = tfv_val.transform(values)

    return idx, tags, attrs, values


print "getting train matrices..."
train_idx, train_tags, train_attrs, train_values, y, tfv_tag, \
           tfv_attr, tfv_val = get_train_matrix(PATH_TO_TRAIN)
train_df = pd.read_csv("datasets/trainData.csv").fillna("")
train_title = train_df.title.values
train_text = train_df.text.values

tfv_title = TfidfVectorizer(min_df=5, max_features=10000, strip_accents='unicode',
                            analyzer='word', token_pattern=r'\w{2,}',
                            ngram_range=(1, 2), use_idf=1, smooth_idf=1,
                            sublinear_tf=1, stop_words=STOPWORDS)
tfv_text = TfidfVectorizer(min_df=5, max_features=50000, strip_accents='unicode',
                           analyzer='word', token_pattern=r'\w{2,}',
                           ngram_range=(1, 1), use_idf=1, smooth_idf=1,
                           sublinear_tf=1, stop_words=STOPWORDS)

train_title = tfv_title.fit_transform(train_title)
train_text = tfv_text.fit_transform(train_text)

print "saving train matrices..."
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
with open("datasets/trainValueSparse.pkl", "w") as f:
    pickle.dump(train_values, f)
    f.close()
with open("datasets/trainTitleSparse.pkl", "w") as f:
    pickle.dump(train_title, f)
    f.close()
with open("datasets/trainTextSparse.pkl", "w") as f:
    pickle.dump(train_text, f)
    f.close()

# collect garbage
train_tags = train_attrs = train_values = None

print "getting test matrices..."
test_idx, test_tags, test_attrs, test_values = \
          get_test_matrix(PATH_TO_TEST, tfv_tag, tfv_attr, tfv_val)
test_df = pd.read_csv("datasets/testData.csv").fillna("")
test_title = test_df.title.values
test_text = test_df.text.values

test_title = tfv_title.transform(test_title)
test_text = tfv_text.transform(test_text)

print "saving test matrices..."
with open("datasets/test_idx.pkl", "w") as f:
    pickle.dump(test_idx, f)
    f.close()
with open("datasets/testTagSparse.pkl", "w") as f:
    pickle.dump(test_tags, f)
    f.close()
with open("datasets/testAttrSparse.pkl", "w") as f:
    pickle.dump(test_attrs, f)
    f.close()
with open("datasets/testValueSparse.pkl", "w") as f:
    pickle.dump(test_values, f)
    f.close()
with open("datasets/testTitleSparse.pkl", "w") as f:
    pickle.dump(test_title, f)
    f.close()
with open("datasets/testTextSparse.pkl", "w") as f:
    pickle.dump(test_text, f)
    f.close()


