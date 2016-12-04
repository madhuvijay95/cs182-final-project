import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import cross_validation
from sklearn.metrics import classification_report, accuracy_score
from knn import kNearestNeighbors
import sys
import time

# import data
filename = 'augmented.csv'
df = pd.read_csv(filename)

# lower case
df.article_title = [x.lower() for x in df.article_title]

# randomly sample fraction of data for testing
df = df.sample(frac=0.1)

# y vector of labels (clickbait or not in our case)
y = list(df.clickbait)

# vectorize the data using tf-idf
vectorizer = TfidfVectorizer(max_features=200, stop_words='english', use_idf=True)
x = np.asarray(vectorizer.fit_transform(list(df.article_title)).todense())

# split data into test and train sets
x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=0.3, random_state=1)

print '================================= hand kNN RESULTS ================================='

knn = kNearestNeighbors()
knn.fit_score(x_train, x_test, y_train, y_test)

sys.stdout.flush()