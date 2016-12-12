import pandas as pd
from knn import kNearestNeighbors
import sys
import time

# import data
trainfile = 'train.csv'
testfile = 'test.csv'

# create dataframes 
train = pd.read_csv(trainfile)
test = pd.read_csv(testfile)

# lower case
train.article_title = [x.lower() for x in train.article_title]
test.article_title = [x.lower() for x in test.article_title]

# randomly sample fraction of data for testing
# df = df.sample(frac=0.2)

print '====================== hand kNN RESULTS ======================'

knn = kNearestNeighbors(k=9, n_features=100000)
knn.fit_score(train, test)

# cross-validation
# knn.cv(train, test, k_vals=[1,3,5,7,9])
# knn.cv(train, test, n_vals=[10,100,1000,10000,100000,1000000])

sys.stdout.flush()