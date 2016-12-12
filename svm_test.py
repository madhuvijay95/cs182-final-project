import pandas as pd
from svm import SupportVectorMachine
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

# format classes into 1, -1 to accommodate signed scoring approach of svm
train.ix[train.clickbait == 0, 'clickbait'] = -1
test.ix[test.clickbait == 0, 'clickbait'] = -1

print '====================== hand SVM RESULTS ======================'

svm = SupportVectorMachine(lmbda=1., n_features=100)
svm.fit_score(train, test)