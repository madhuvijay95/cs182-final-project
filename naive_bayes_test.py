import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer#, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from naive_bayes import NaiveBayes
import matplotlib.pyplot as plt

filename = 'augmented.csv'
augmented_df = pd.read_csv(filename)
print 'Working on file: %s' % filename

vectorizer = CountVectorizer(max_df=0.1)#, min_df=5, ngram_range=(1,3))
X = vectorizer.fit_transform(augmented_df['article_title'])
vocab_rev = {v:k for k,v in vectorizer.vocabulary_.items()}
y = np.array(augmented_df['clickbait'])
print 'Shape of counts matrix:', X.shape
print

train_mask = np.random.rand(X.shape[0]) <= 0.7 # TODO doesn't necessarily split exactly 70-30
X_train = X[train_mask]
X_test = X[~train_mask]
y_train = y[train_mask]
y_test = y[~train_mask]

nb = MultinomialNB()
nb.fit(X_train, y_train)
print 'sklearn NB test accuracy: %.5f' % nb.score(X_test, y_test)

nb_madhu = NaiveBayes()
nb_madhu.fit(X_train, y_train)
print 'Madhu NB test accuracy: %.5f' % nb_madhu.score(X_test, y_test)
print 'Madhu NB test accuracy (alt version): %.5f' % nb_madhu.score2(X_test, y_test)
print

alphas = [0, 0.01, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 5]
scores = nb_madhu.cross_validation(X_train, y_train, alphas)
scores = {k:np.mean(v) for k,v in scores.items()}
print 'Mean cross-validation accuracy for each alpha:', scores
plt.plot(*(zip(*(sorted(scores.items(), key = lambda tup : tup[0])))))
plt.show()
alpha = max(scores, key = lambda k : scores[k])
print 'Best alpha:', alpha
nb_madhu.fit(X_train, y_train, alpha=alpha, vocab=vectorizer.vocabulary_)
print 'Madhu NB test accuracy (with optimal alpha): %.5f' % nb_madhu.score(X_test, y_test)
print

print 'Representative words:'
print nb_madhu.representative_words()
