import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from naive_bayes import NaiveBayes, NotNaiveBayes
from svm import SupportVectorMachine
from knn import kNearestNeighbors
import numpy as np
import sys
from sklearn.metrics import classification_report, accuracy_score

# import data into dataframes
trainfile = 'train.csv'
testfile = 'test.csv'
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
# remove apostrophes
train['article_title'] = [title.replace('\'', '') for title in train['article_title']]
test['article_title'] = [title.replace('\'', '') for title in test['article_title']]

################################################ NAIVE BAYES ################################################

vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,3))
vectorizer.fit(train['article_title'])
vocab = vectorizer.vocabulary_

# create word count matrices using the CountVectorizer object
X_train = vectorizer.transform(train['article_title'])
X_test = vectorizer.transform(test['article_title'])

# create vectors of class labels
y_train = np.array(train['clickbait'])
y_test = np.array(test['clickbait'])

# initialize Naive Bayes model
naive_bayes = NaiveBayes()

# choose best value for hyperparameter alpha (from cross-validation)
alpha = 0.5
naive_bayes.fit(X_train, y_train, vocab, alpha=alpha)
predictions = naive_bayes.predict(X_test)
print classification_report(y_test, predictions, target_names = ['clickbait','non-clickbait'])
print 'Naive Bayes test accuracy (with optimal alpha): %.5f' % naive_bayes.score(X_test, y_test)
print
sys.stdout.flush()

# output lists of representative words for each topic
print 'Representative words:'
rep_words = naive_bayes.representative_words(n_words=50)
print 'Non-clickbait:', reduce(lambda a,b : a + ', ' + b, rep_words[0])
print 'Clickbait:', reduce(lambda a,b : a + ', ' + b, rep_words[1])
print
sys.stdout.flush()




################################################ SVM ################################################

train.ix[train.clickbait == 0, 'clickbait'] = -1
test.ix[test.clickbait == 0, 'clickbait'] = -1

print '====================== hand SVM RESULTS ======================'
#svm = SupportVectorMachine(lmbda=1., n_features=100)
#svm.fit_score(train, test)
#sys.stdout.flush()

################################################ kNN ################################################

train = pd.read_csv(trainfile)
test = pd.read_csv(testfile)
train.article_title = [x.lower() for x in train.article_title]
test.article_title = [x.lower() for x in test.article_title]
print '====================== hand kNN RESULTS ======================'
knn = kNearestNeighbors(k=3, n_features=10)
knn.fit_score(train, test)