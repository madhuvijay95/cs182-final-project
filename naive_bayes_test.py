import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer#, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from naive_bayes import NaiveBayes, NotNaiveBayes
import sys
import gensim
import cPickle as pickle
from scipy.sparse import hstack, csr_matrix
import matplotlib.pyplot as plt

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
vectorizer = CountVectorizer(stop_words='english', min_df=5, max_df=0.1, ngram_range=(1,3))
X_train = vectorizer.fit_transform(train_df['article_title'])
vocab = vectorizer.vocabulary_
vocab_rev = {v:k for k,v in vectorizer.vocabulary_.items()}
y_train = np.array(train_df['clickbait'])

X_test = vectorizer.transform(test_df['article_title'])
y_test = np.array(test_df['clickbait'])


naive_bayes = NaiveBayes()
alphas = [0, 0.01, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5]
scores = naive_bayes.cross_validation(X_train, y_train, alphas, output=True)
with open('naive_bayes_cross-validation_scores.p', 'w') as f:
    pickle.dump(scores, f)
scores = {k:np.mean(v) for k,v in scores.items()}
plt.plot(alphas, [scores[alpha] for alpha in alphas])
plt.title('Naive Bayes Cross-Validation')
plt.xlabel('Alpha')
plt.ylabel('Mean cross-validation accuracy')
plt.savefig('naive_bayes_cross-validation.png')
plt.show()
alpha = max(scores, key = lambda k : scores[k])
print 'Best alpha:', alpha
naive_bayes.fit(X_train, y_train, alpha=alpha, vocab=vocab)
print 'Madhu NB test accuracy (with optimal alpha): %.5f' % naive_bayes.score(X_test, y_test)
print

print 'Representative words:'
print naive_bayes.representative_words(n_words=20)
print

rep_errors = naive_bayes.representative_errors(X_test, y_test)
print 'Some non-clickbait titles that were misclassified as clickbait:'
for title in test_df['article_title'].iloc[rep_errors[0]]:
    print title.encode('utf-8')
print
print 'Some clickbait titles that were misclassified as non-clickbait:'
for title in test_df['article_title'].iloc[rep_errors[1]]:
    print title.encode('utf-8')

print
print
sys.stdout.flush()



##### LDA STUFF
#num_topics = 3
#X_train_lda = [[(i, count) for i, count in enumerate(np.array(row.todense())[0]) if count > 0] for row in X_train]
#X_test_lda = [[(i, count) for i, count in enumerate(np.array(row.todense())[0]) if count > 0] for row in X_test]
#print 'finished transforming X for LDA'
#sys.stdout.flush()
#lda = gensim.models.ldamodel.LdaModel(corpus=X_train_lda, num_topics=num_topics, id2word=vocab_rev)
#print lda.print_topics(num_words=20)
#print
#sys.stdout.flush()
#vocab = dict(vectorizer.vocabulary_.items() + [('LDA Topic %d' % i, len(vectorizer.vocabulary_) + i) for i in range(num_topics)])
#vocab_rev = {v:k for k,v in vectorizer.vocabulary_.items()}
#n_append = 10
#print X_train.shape, X_test.shape
#sys.stdout.flush()
#X_train_lda_output = csr_matrix(np.array([list(zip(*(lda[row]))[1]) for row in X_train_lda]))
#X_test_lda_output = csr_matrix(np.array([list(zip(*(lda[row]))[1]) for row in X_test_lda]))
#for _ in range(n_append):
#    X_train = hstack((X_train, X_train_lda_output), format='csr')
#    X_test = hstack((X_test, X_test_lda_output), format='csr')
#    print X_train.shape, X_test.shape
#    sys.stdout.flush()



print '=================================LESS-NAIVE BAYES RESULTS================================='

not_naive_bayes = NotNaiveBayes()
X_train_new = not_naive_bayes.convert(train_df['article_title'], vectorizer)
X_test_new = not_naive_bayes.convert(test_df['article_title'], vectorizer)

alphas = [0, 0.01, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5]
scores = not_naive_bayes.cross_validation(X_train_new, y_train, alphas, output=True)
with open('not-naive_bayes_cross-validation_scores.p', 'w') as f:
    pickle.dump(scores, f)
scores = {k:np.mean(v) for k,v in scores.items()}
plt.plot(alphas, [scores[alpha] for alpha in alphas])
plt.title('Not-Naive Bayes Cross-Validation')
plt.xlabel('Alpha')
plt.ylabel('Mean cross-validation accuracy')
plt.savefig('not-naive_bayes_cross-validation.png')
plt.show()
alpha = max(scores, key = lambda k : scores[k])
print 'Best alpha:', alpha
not_naive_bayes.fit(X_train_new, y_train, alpha=alpha, vocab=vocab)
print 'Not-Naive Bayes test accuracy (with optimal alpha): %.5f' % not_naive_bayes.score(X_test_new, y_test)
print
