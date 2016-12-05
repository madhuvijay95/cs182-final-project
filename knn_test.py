import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
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

knn = kNearestNeighbors()
knn.fit_score(train, test, k=3, n_features=200)

# cross-validation
# knn.cv(df, k_vals=[1,3,5,7,9])
# knn.cv(df, n_vals=[10,50,100,250,500])

sys.stdout.flush()