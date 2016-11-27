import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer#, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from naive_bayes import NaiveBayes, NotNaiveBayes
import sys

filename = 'augmented.csv'
augmented_df = pd.read_csv(filename)
print 'Working on file: %s' % filename
print augmented_df.head()
print
print


print '=================================NAIVE BAYES RESULTS================================='

vectorizer = CountVectorizer(max_df=0.1, min_df=5)#, ngram_range=(1,3))
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
#plt.plot(*(zip(*(sorted(scores.items(), key = lambda tup : tup[0])))))
#plt.show()
alpha = max(scores, key = lambda k : scores[k])
print 'Best alpha:', alpha
nb_madhu.fit(X_train, y_train, alpha=alpha, vocab=vectorizer.vocabulary_)
print 'Madhu NB test accuracy (with optimal alpha): %.5f' % nb_madhu.score(X_test, y_test)
print

print 'Representative words:'
print nb_madhu.representative_words()
print

rep_errors = nb_madhu.representative_errors(X_test, y_test)
print 'Some non-clickbait titles that were misclassified as clickbait:'
for title in augmented_df[~train_mask]['article_title'].iloc[rep_errors[0]]:
    print title
print
print 'Some clickbait titles that were misclassified as non-clickbait:'
for title in augmented_df[~train_mask]['article_title'].iloc[rep_errors[1]]:
    print title

print
print
sys.stdout.flush()


print '=================================LESS-NAIVE BAYES RESULTS================================='

nb_madhu = NotNaiveBayes()
X_train_new = nb_madhu.convert(augmented_df[train_mask]['article_title'], vectorizer)
X_test_new = nb_madhu.convert(augmented_df[~train_mask]['article_title'], vectorizer)
nb_madhu.fit(X_train_new, y_train, vocab=vectorizer.vocabulary_, alpha=0.001)
for _ in range(5):
    print nb_madhu.generate(0)
    sys.stdout.flush()
print
for _ in range(5):
    print nb_madhu.generate(1)
    sys.stdout.flush()
#for alpha in [0.01, 0.05, 0.1, 0.5, 1]:
#    nb_madhu.fit(X_train_new, y_train, alpha=alpha)
#    print 'Score (with alpha=%.2f): %.5f' % (alpha, nb_madhu.score(X_test_new, y_test))
#    sys.stdout.flush()
