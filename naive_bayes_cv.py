import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from naive_bayes import NaiveBayes, NotNaiveBayes
import sys
import gensim
import cPickle as pickle
from scipy.sparse import hstack, csr_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# import data into dataframes
trainfile = 'train.csv'
testfile = 'test.csv'
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
# remove apostrophes
train_df['article_title'] = [title.replace('\'', '') for title in train_df['article_title']]
test_df['article_title'] = [title.replace('\'', '') for title in test_df['article_title']]

# use sklearn's CountVectorizer to determine the vocabulary and create a vectorizer object
vectorizer = CountVectorizer(stop_words='english', ngram_range=(1,3))
vectorizer.fit(train_df['article_title'])
vocab = vectorizer.vocabulary_
vocab_rev = {v:k for k,v in vectorizer.vocabulary_.items()}

# create word count matrices using the CountVectorizer object
X_train = vectorizer.transform(train_df['article_title'])
X_test = vectorizer.transform(test_df['article_title'])

# create vectors of class labels
y_train = np.array(train_df['clickbait'])
y_test = np.array(test_df['clickbait'])

# use TF-IDF (term frequency-inverse document frequency) to transform the count matrices, downweighting very common
# words and upweighting more rare useful ones
tfidf = TfidfTransformer()
tfidf.fit(X_train)
X_train_tfidf = tfidf.transform(X_train)
X_test_tfidf = tfidf.transform(X_test)



# list of smoothing parameters to test in cross-validation
alphas = [0, 0.01, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5, 10, 20, 50]
# number of subsets to use for cross-validation
n_folds = 5

# initialize Naive Bayes model
naive_bayes = NaiveBayes()


print '=============================NAIVE BAYES CROSS-VALIDATION RESULTS (WITH TF-IDF)============================='

# compute and store cross-validation scores (using count matrices with TF-IDF transformation)
scores_tfidf = naive_bayes.cross_validation(X_train_tfidf, y_train, vocab, alphas, output=True, k=n_folds)
with open('naive_bayes_cross-validation_scores_tfidf.p', 'w') as f:
    pickle.dump(scores_tfidf, f)
scores_tfidf = {k:np.mean(v) for k,v in scores_tfidf.items()}
print 'Best alpha:', max(scores_tfidf, key = lambda k : scores_tfidf[k])
sys.stdout.flush()

# plot cross-validation scores (with TF-IDF)
plt.plot(alphas, [scores_tfidf[alpha] for alpha in alphas])
plt.title('Naive Bayes Cross-Validation (TF-IDF)')
plt.xlabel('Alpha')
plt.ylabel('Mean cross-validation accuracy')
plt.savefig('naive_bayes_cross-validation_tfidf.png')
plt.show()
plt.plot(alphas[0:8], [scores_tfidf[alpha] for alpha in alphas[0:8]])
plt.title('Naive Bayes Cross-Validation (TF-IDF)')
plt.xlabel('Alpha')
plt.ylabel('Mean cross-validation accuracy')
plt.savefig('naive_bayes_cross-validation_tfidf_zoomed.png')
plt.show()
print


print '=============================NAIVE BAYES CROSS-VALIDATION RESULTS (WITHOUT TF-IDF)============================='

# compute and store cross-validation scores (using count matrices without TF-IDF transformation)
scores_no_tfidf = naive_bayes.cross_validation(X_train, y_train, vocab, alphas, output=True, k=n_folds)
with open('naive_bayes_cross-validation_scores.p', 'w') as f:
    pickle.dump(scores_no_tfidf, f)
scores_no_tfidf = {k:np.mean(v) for k,v in scores_no_tfidf.items()}
print 'Best alpha:', max(scores_no_tfidf, key = lambda k : scores_no_tfidf[k])
sys.stdout.flush()

# plot cross-validation scores (without TF-IDF)
plt.plot(alphas, [scores_no_tfidf[alpha] for alpha in alphas])
plt.title('Naive Bayes Cross-Validation')
plt.xlabel('Alpha')
plt.ylabel('Mean cross-validation accuracy')
plt.savefig('naive_bayes_cross-validation.png')
plt.show()
plt.plot(alphas[0:8], [scores_no_tfidf[alpha] for alpha in alphas[0:8]])
plt.title('Naive Bayes Cross-Validation')
plt.xlabel('Alpha')
plt.ylabel('Mean cross-validation accuracy')
plt.savefig('naive_bayes_cross-validation_zoomed.png')
plt.show()
print

# check whether performance was better with or without TF-IDF
if max(scores_tfidf.values()) >= max(scores_no_tfidf.values()):
    print 'Optimal performance was better with TF-IDF (%.3f) than without it (%.3f).' %\
          (max(scores_tfidf.values()), max(scores_no_tfidf.values()))
    scores = scores_tfidf
    X_train_final = X_train_tfidf
    X_test_final = X_test_tfidf
else:
    print 'Optimal performance was better without TF-IDF (%.3f) than with it (%.3f).' %\
          (max(scores_no_tfidf.values()), max(scores_tfidf.values()))
    scores = scores_no_tfidf
    X_train_final = X_train
    X_test_final = X_test

# choose best value for hyperparameter alpha, and fit/test the model with that value
alpha = max(scores, key = lambda k : scores[k])
print 'Best alpha:', alpha
sys.stdout.flush()
naive_bayes.fit(X_train_final, y_train, vocab, alpha=alpha)
print 'Naive Bayes test accuracy (with optimal alpha): %.5f' % naive_bayes.score(X_test_final, y_test)
print
sys.stdout.flush()

# output lists of representative words for each topic
print 'Representative words:'
for lst in naive_bayes.representative_words(n_words=20):
    print lst
print
sys.stdout.flush()

# print some examples of topics that were misclassified (in both directions)
rep_errors = naive_bayes.representative_errors(X_test_final, y_test)
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



# This section performs a very limited test of one of the ideas mentioned in the "Discussion" section of our report, to
# perform unsupervised clustering (using gensim's LDA model)
print '=================================NAIVE BAYES RESULTS (WITH LDA FEATURES)================================='

# cluster data into 3 topics using LDA
num_topics = 3
print 'Working on transforming data matrix to use as input for LDA...',
sys.stdout.flush()
X_train_lda = [[(i, count) for i, count in enumerate(np.array(row.todense())[0]) if count > 0] for row in X_train_final]
X_test_lda = [[(i, count) for i, count in enumerate(np.array(row.todense())[0]) if count > 0] for row in X_test_final]
print 'finished'
sys.stdout.flush()
lda = gensim.models.ldamodel.LdaModel(corpus=X_train_lda, num_topics=num_topics, id2word=vocab_rev)

# print words for LDA topics
print 'LDA topics:'
print lda.print_topics(num_words=20)
print
sys.stdout.flush()

# construct new vocabulary dictionary with extra terms for the LDA topics
vocab = dict(vectorizer.vocabulary_.items() + [('LDA Topic %d' % i, len(vectorizer.vocabulary_) + i) for i in range(num_topics)])

# append the LDA cluster probabilities to the right side of the feature matrix, and re-run the naive Bayes classifier
X_train_lda_output = csr_matrix(np.array([list(zip(*(lda[row]))[1]) for row in X_train_lda]))
X_test_lda_output = csr_matrix(np.array([list(zip(*(lda[row]))[1]) for row in X_test_lda]))
X_train_final = hstack((X_train_final, X_train_lda_output), format='csr')
X_test_final = hstack((X_test_final, X_test_lda_output), format='csr')
naive_bayes.fit(X_train_final, y_train, vocab, alpha=alpha)
print 'Naive Bayes test accuracy (with optimal alpha, and with LDA appended): %.5f' %\
      naive_bayes.score(X_test_final, y_test)
sys.stdout.flush()
print
print
sys.stdout.flush()



print '=================================LESS-NAIVE BAYES RESULTS================================='

# create word count matrices using the CountVectorizer object
vectorizer = CountVectorizer(stop_words='english')
vectorizer.fit(train_df['article_title'])
vocab = vectorizer.vocabulary_

# initialize Naive Bayes model
not_naive_bayes = NotNaiveBayes()
# use convert function to transform data matrices into correct format for not-naive Bayes model
X_train_new = not_naive_bayes.convert(train_df['article_title'], vectorizer)
X_test_new = not_naive_bayes.convert(test_df['article_title'], vectorizer)

# list of smoothing parameters to test in cross-validation
alphas = [0, 0.01, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5, 10, 20, 50]

# compute and store cross-validation scores
scores = not_naive_bayes.cross_validation(X_train_new, y_train, vocab, alphas, output=True, k=n_folds)
with open('not-naive_bayes_cross-validation_scores.p', 'w') as f:
    pickle.dump(scores, f)
scores = {k:np.mean(v) for k,v in scores.items()}

# plot cross-validation scores
plt.plot(alphas, [scores[alpha] for alpha in alphas])
plt.title('Not-Naive Bayes Cross-Validation')
plt.xlabel('Alpha')
plt.ylabel('Mean cross-validation accuracy')
plt.savefig('not-naive_bayes_cross-validation.png')
plt.show()
plt.plot(alphas[0:8], [scores[alpha] for alpha in alphas[0:8]])
plt.title('Not-Naive Bayes Cross-Validation')
plt.xlabel('Alpha')
plt.ylabel('Mean cross-validation accuracy')
plt.savefig('not-naive_bayes_cross-validation_zoomed.png')
plt.show()

# choose best value for hyperparameter alpha, and fit/test the model with that value
alpha = max(scores, key = lambda k : scores[k])
print 'Best alpha:', alpha
sys.stdout.flush()
not_naive_bayes.fit(X_train_new, y_train, vocab, alpha=alpha)
print 'Not-Naive Bayes test accuracy (with optimal alpha): %.5f' % not_naive_bayes.score(X_test_new, y_test)
print
sys.stdout.flush()

# print some examples of topics that were misclassified (in both directions)
rep_errors = not_naive_bayes.representative_errors(X_test_new, y_test)
print 'Some non-clickbait titles that were misclassified as clickbait:'
for title in test_df['article_title'].iloc[rep_errors[0]]:
    print title.encode('utf-8')
print
print 'Some clickbait titles that were misclassified as non-clickbait:'
for title in test_df['article_title'].iloc[rep_errors[1]]:
    print title.encode('utf-8')
