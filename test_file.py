import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from naive_bayes import NaiveBayes, NotNaiveBayes
from svm import SupportVectorMachine
import numpy as np
import sys
from sklearn.metrics import classification_report, accuracy_score

# import data into dataframes
trainfile = 'train.csv'
testfile = 'test.csv'
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

################################################ NAIVE BAYES ################################################

# use sklearn's CountVectorizer to determine the vocabulary and create a vectorizer object
vectorizer = CountVectorizer(stop_words='english', ngram_range=(1,3))
vectorizer.fit(train['article_title'])
vocab = vectorizer.vocabulary_
vocab_rev = {v:k for k,v in vectorizer.vocabulary_.items()}

# create word count matrices using the CountVectorizer object
X_train = vectorizer.transform(train['article_title'])
X_test = vectorizer.transform(test['article_title'])

# create vectors of class labels
y_train = np.array(train['clickbait'])
y_test = np.array(test['clickbait'])

# use TF-IDF (term frequency-inverse document frequency) to transform the count matrices, downweighting very common
# words and upweighting more rare useful ones
tfidf = TfidfTransformer()
tfidf.fit(X_train)
X_train_tfidf = tfidf.transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# initialize Naive Bayes model
naive_bayes = NaiveBayes()

# choose best value for hyperparameter alpha, and fit/test the model with that value
alpha = 0.5
naive_bayes.fit(X_train_tfidf, y_train, vocab, alpha=alpha)
predictions = naive_bayes.predict(X_test_tfidf)
print classification_report(y_test, predictions, target_names = ['clickbait','non-clickbait'])
print 'Naive Bayes test accuracy (with optimal alpha): %.5f' % naive_bayes.score(X_test_tfidf, y_test)
print
sys.stdout.flush()





################################################ SVM ################################################

train.ix[train.clickbait == 0, 'clickbait'] = -1
test.ix[test.clickbait == 0, 'clickbait'] = -1

print '====================== hand SVM RESULTS ======================'

svm = SupportVectorMachine(lmbda=1., n_features=100)
svm.fit_score(train, test)