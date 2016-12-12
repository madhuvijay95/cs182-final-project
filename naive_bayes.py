import numpy as np
import scipy as sp
from scipy import sparse
import sys

# Implements the naive Bayes model, which assumes that each word in each document depends only on the class assignment
# (and is conditionally independent of all other words in the document, given the class).
class NaiveBayes:
    # Fits the Naive Bayes model, given a matrix X of word counts (e.g. using the output of sklearn's CountVectorizer),
    # a list y of class assignments, and a vocabulary.
    def fit(self, X, y, vocab, alpha=1.):
        # store parameters
        self.alpha = alpha
        self.vocab = vocab
        self.vocab_rev = {v:k for k,v in vocab.items()} if vocab is not None else None

        # store information on classes passed in
        self.classes = list(set(y))
        self.classes_rev = {v:k for k,v in enumerate(self.classes)}
        self.nclasses = len(self.classes)

        # compute prior distribution over classes, using counts for each class
        self.classcounts = np.array([float((np.array(y)==c).sum()) for c in self.classes])
        self.classprobs = self.classcounts / self.classcounts.sum()
        self.logclassprobs = np.log(self.classprobs)

        # compute the number of times that each word shows up in each topic
        self.wordcounts = np.array([map(float, np.array(X[y==c].sum(axis=0))[0]) for c in self.classes])
        # augment by Dirichlet parameter alpha (for smoothing), and normalize the result to have sum 1 for each topic
        self.wordprobs = ((self.wordcounts + alpha).T / (self.wordcounts + alpha).sum(axis=1)).T
        # compute log probability of each word for each topic
        self.logwordprobs = np.log(self.wordprobs)

    # Take in an N x D data matrix X and output an N x K matrix (where K is the number of topics) containing the
    # posterior log probability of each topic assignment for each data point.
    def predict_log_proba(self, X):
        # compute the log likelihood of each document given each possible class assignment
        log_proba = X.dot(self.logwordprobs.T)
        # incorporate prior class probabilities to find posterior probability of each class for each data point
        log_proba += self.logclassprobs
        # normalize each row to sum to 1
        log_proba = (log_proba.T - np.log(np.exp(log_proba).sum(axis=1))).T
        return log_proba

    # Same as predict_log_proba, but outputs probabilities instead of log probabilities
    def predict_proba(self, X):
        return np.exp(self.predict_log_proba(X))

    # Same as predict_log_proba and predict_proba, but outputs the single most likely class assignment (instead of a
    # full distribution) for each data point.
    def predict(self, X):
        # compute log probability of each class for each data point
        log_proba = self.predict_log_proba(X)
        # compute and return the most likely class for each row of X
        predictions = np.argmax(log_proba, axis=1)
        predictions = [self.classes[i] for i in predictions]
        return predictions

    # Computes accuracy of the classifier on the given data (i.e. computes predictions on X, and computes how frequently
    # the predictions match the actual class assignments).
    def score(self, X, y):
        predictions = self.predict(X)
        n_correct = sum([p==c for p, c in zip(predictions, y)])
        return float(n_correct) / X.shape[0]

    # Use k-fold cross-validation to test different values of the smoothing parameter alpha.
    def cross_validation(self, X, y, vocab, alphas, k=5, output=False):
        # compute number of data points
        n_samples = X.shape[0]
        assert(len(y) == n_samples)
        # generate a random permutation of the data points
        indices = range(n_samples)
        np.random.shuffle(indices)
        # split the indices into k subsets, and split X and y accordingly
        split = [sorted(indices[start::k]) for start in range(k)]
        split_X = [X[ind_list] for ind_list in split]
        split_y = [np.array(y)[ind_list] for ind_list in split]

        # initialize a dictionary to store accuracy rates
        scores = dict()
        # loop through all allowed values of alpha
        for alpha in alphas:
            if output:
                print 'Cross-validation on alpha=%.3f:  ' % alpha,
                sys.stdout.flush()
            # initialize scores[alpha] as a list to store the k accuracy rates
            scores[alpha] = []
            # loop over the k subsets to leave out
            for leave_out in range(k):
                # train the model on all data except the left-out set
                X_train = sp.sparse.vstack(tuple(split_X[i] for i in range(k) if i != leave_out))
                y_train = np.concatenate(tuple(split_y[i] for i in range(k) if i != leave_out))
                self.fit(X_train, y_train, vocab, alpha=alpha)
                # compute and store the accuracy on the held-out validation subset
                score = self.score(split_X[leave_out], split_y[leave_out])
                scores[alpha].append(score)
                if output:
                    print '%.3f' % score,
                    sys.stdout.flush()
            if output:
                print ' ==> %.3f' % np.mean(scores[alpha])
                sys.stdout.flush()
        return scores

    # Return a list of "most representative" words (i.e. words for which the log probability of that word in the given
    # topic is large relative to the average log probability of the word given other topics) for each topic
    def representative_words(self, n_words=10):
        # for each word, compute the average of log[p(word|class)] across all classes
        logwordprobs_avg = np.mean(self.logwordprobs, axis=0)
        # use logwordprobs_avg to normalize the p(word|class) matrix
        logwordprobs_normalized = self.logwordprobs - logwordprobs_avg
        # compute indices of the most representative words (i.e. the highest values in the matrix above) for each topic
        indices = [sorted(range(len(lst)), key = lambda ind : -lst[ind])[0:n_words] for lst in logwordprobs_normalized]
        # return the indices, or lists of words if the vocabulary is known
        if self.vocab_rev is None:
            return indices
        else:
            return [[self.vocab_rev[ind] for ind in lst] for lst in indices]

    # Return indices of the most egregious errors for each topic (i.e. for each class c, return a list of indices of
    # data points in class c for which the classifier thought p(c|data) was very low).
    def representative_errors(self, X, y, n_examples=10):
        # use predict_proba to compute p(class|data) for each data point and each class
        proba = self.predict_proba(X)
        # find the predictive probability p(correct class|data) for each data point
        posterior = np.array([lst[self.classes_rev[c]] for lst, c in zip(proba, y)])
        # split up the posterior array by class
        posterior_split = [posterior[y == self.classes_rev[c]] for c in self.classes]
        # find and return the indices of the most egregious errors within each class
        extreme_errors = [sorted(range(len(lst)), key = lambda ind : lst[ind])[0:n_examples] for lst in posterior_split]
        class_indices = [np.arange(X.shape[0])[y == self.classes_rev[c]] for c in self.classes]
        extreme_errors = [[ind_lst[err_ind] for err_ind in err_lst] for err_lst, ind_lst in zip(extreme_errors, class_indices)]
        return extreme_errors


# Implements the not-naive Bayes model, which assumes that each word in each document depends on the class assignment
# and on the immediately preceding word.
class NotNaiveBayes:
    # Fit model, given X (which is a list of lists, where each sub-list represents a list of vocabulary indices of
    # words in a given title, in order), a list y of class assignments, and a vocabulary.
    def fit(self, X, y, vocab, alpha=1.):
        # store parameters
        self.alpha = alpha
        self.vocab = vocab
        self.vocab_rev = {v:k for k,v in vocab.items()} if vocab is not None else None
        self.vocab_size = len(vocab) if vocab is not None else max([max(lst) for lst in X if len(lst) > 0])+1

        # store information on classes passed in
        self.classes = list(set(y))
        self.classes_rev = {v:k for k,v in enumerate(self.classes)}
        self.nclasses = len(self.classes)

        # compute prior distribution over classes, using counts for each class
        self.classcounts = np.array([float((np.array(y)==c).sum()) for c in self.classes])
        self.classprobs = self.classcounts / self.classcounts.sum()
        self.logclassprobs = np.log(self.classprobs)

        # have a list of transition matrices (one for each class)
        self.count_mats = []
        # loop through classes
        for c in self.classes:
            # Initialize sparse transition matrix. Each index from 0..vocab_size-1 represents the corresponding element
            # of the vocabulary; the last row represents the distribution for the first word of each title.
            cb_mat = sp.sparse.lil_matrix((self.vocab_size+1, self.vocab_size))
            # loop through all data points that are in the current class c
            for lst in [x for x, k in zip(X, y) if k == c]:
                # pad the start of the word list with an extra value to represent the start
                lst_new = [cb_mat.shape[0]-1] + lst
                # list of tuples of consecutive word indices
                seq_lst = zip(lst_new[0:-1], lst_new[1:])
                # increment each element of the count matrix appropriately
                for tup in seq_lst:
                    cb_mat[tup] += 1
            self.count_mats.append(cb_mat.copy())

    # Helper function: Takes a dataset (in the form of a list, array, or Series of documents) and a CountVectorizer
    # object, and converts the dataset into the correct form of an input to this algorithm.
    def convert(self, dataset, countvectorizer):
        analyzer = countvectorizer.build_analyzer()
        return [[countvectorizer.vocabulary_[word] for word in analyzer(title) if word in countvectorizer.vocabulary_]
                for title in dataset]

    # Output an N x K matrix (where N is the number of data points and K is the number of topics) containing the
    # posterior log probability of each topic assignment for each data point.
    def predict_log_proba(self, X):
        # pad the start of each word list with an extra value to represent the start
        X_modified = [[self.vocab_size] + lst for lst in X]
        # convert each list into a list of tuples of consecutive elements
        X_modified = [zip(lst[0:-1], lst[1:]) for lst in X_modified]
        # compute the log likelihood of each document given each possible class assignment
        log_proba = np.array([[sum([np.log(float(mat[tup[0], tup[1]] + self.alpha) / (mat[tup[0]].sum() + self.alpha * mat.shape[1]))
                                    for tup in lst])
                               for mat in self.count_mats]
                              for lst in X_modified])
        # incorporate prior class probabilities to find posterior probability of each class for each data point
        log_proba += self.logclassprobs
        # normalize each row to sum to 1
        log_proba = (log_proba.T - np.log(np.exp(log_proba).sum(axis=1))).T
        return log_proba

    # Same as predict_log_proba, but outputs probabilities instead of log probabilities
    def predict_proba(self, X):
        return np.exp(self.predict_log_proba(X))

    # Same as predict_log_proba and predict_proba, but outputs the single most likely class assignment (instead of a
    # full distribution) for each data point.
    def predict(self, X):
        # compute log probability of each class for each data point
        log_proba = self.predict_log_proba(X)
        # compute and return the most likely class for each data point in X
        predictions = np.argmax(log_proba, axis=1)
        predictions = [self.classes[i] for i in predictions]
        return predictions

    # Computes accuracy of the classifier on the given data (i.e. computes predictions on X, and computes how frequently
    # the predictions match the actual class assignments).
    def score(self, X, y):
        predictions = self.predict(X)
        n_correct = sum([p==c for p, c in zip(predictions, y)])
        return float(n_correct) / len(X)

    # Use k-fold cross-validation to test different values of the smoothing parameter alpha.
    def cross_validation(self, X, y, vocab, alphas, k=5, output=False):
        # compute number of data points
        n_samples = len(X)
        assert(len(y) == n_samples)
        # generate a random permutation of the data points
        indices = range(n_samples)
        np.random.shuffle(indices)
        # split the indices into k subsets, and split X and y accordingly
        split = [sorted(indices[start::k]) for start in range(k)]
        split_X = [[X[i] for i in ind_list] for ind_list in split]
        split_y = [np.array(y)[ind_list] for ind_list in split]

        # initialize a dictionary to store accuracy rates
        scores = dict()
        # loop through all allowed values of alpha
        for alpha in alphas:
            if output:
                print 'Cross-validation on alpha=%.3f:  ' % alpha,
                sys.stdout.flush()
            # initialize scores[alpha] as a list to store the k accuracy rates
            scores[alpha] = []
            count_mats_all = []
            # loop over the k subsets (for training)
            for leave_out in range(k):
                # compute and store a matrix of counts for the current subset alone
                self.fit(split_X[leave_out], split_y[leave_out], vocab, alpha=alpha)
                count_mats_all.append(self.count_mats)
            # loop over the k subsets (for testing)
            for leave_out in range(k):
                # compute the count matrix for everything excluding the current subset (by summing matrices from the
                # previous loop)
                self.count_mats = [reduce(lambda m1, m2 : m1+m2, [count_mats_all[i][c] for i in range(k) if i != leave_out])
                                   for c in range(self.nclasses)]
                # compute and store the accuracy on the held-out validation subset
                score = self.score(split_X[leave_out], split_y[leave_out])
                scores[alpha].append(score)
                if output:
                    print '%.3f' % score,
                    sys.stdout.flush()
            if output:
                print ' ==> %.3f' % np.mean(scores[alpha])
                sys.stdout.flush()
        return scores

    # Return indices of the most egregious errors for each topic (i.e. for each class c, return a list of indices of
    # data points in class c for which the classifier thought p(c|data) was very low).
    def representative_errors(self, X, y, n_examples=10):
        # use predict_proba to compute p(class|data) for each data point and each class
        proba = self.predict_proba(X)
        # find the predictive probability p(correct class|data) for each data point
        posterior = np.array([lst[self.classes_rev[c]] for lst, c in zip(proba, y)])
        # split up the posterior array by class
        posterior_split = [posterior[y == self.classes_rev[c]] for c in self.classes]
        # find and return the indices of the most egregious errors within each class
        extreme_errors = [sorted(range(len(lst)), key = lambda ind : lst[ind])[0:n_examples] for lst in posterior_split]
        class_indices = [np.arange(len(X))[y == self.classes_rev[c]] for c in self.classes]
        extreme_errors = [[ind_lst[err_ind] for err_ind in err_lst] for err_lst, ind_lst in zip(extreme_errors, class_indices)]
        return extreme_errors
