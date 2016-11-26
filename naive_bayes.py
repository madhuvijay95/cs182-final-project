import numpy as np
import scipy as sp

class NaiveBayes:
    def fit(self, X, y, alpha=1., vocab=None):
        self.X = np.array(X)
        self.y = np.array(y)
        self.alpha = alpha
        self.vocab = vocab
        self.vocab_rev = {v:k for k,v in vocab.items()} if vocab is not None else None
        self.classes = list(set(self.y))
        self.classes_rev = {v:k for k,v in enumerate(self.classes)}
        self.nclasses = len(self.classes)

        self.classcounts = np.array([float((self.y==c).sum()) for c in self.classes])
        self.classprobs = self.classcounts / self.classcounts.sum()
        self.logclassprobs = np.log(self.classprobs)

        self.wordcounts = np.array([map(float, np.array(X[y==c].sum(axis=0))[0]) for c in self.classes])
        self.wordprobs = ((self.wordcounts + alpha).T / (self.wordcounts + alpha).sum(axis=1)).T
        self.logwordprobs = np.log(self.wordprobs)

    def predict_log_proba(self, X):
        log_proba = X.dot(self.logwordprobs.T)
        log_proba += self.logclassprobs
        log_proba = (log_proba.T - np.log(np.exp(log_proba).sum(axis=1))).T
        return log_proba

    def predict_proba(self, X):
        return np.exp(self.predict_log_proba(X))

    def predict(self, X):
        log_proba = self.predict_log_proba(X)
        predictions = np.argmax(log_proba, axis=1)
        predictions = [self.classes[i] for i in predictions]
        return predictions

    def score(self, X, y):
        predictions = self.predict(X)
        n_correct = sum([p==c for p, c in zip(predictions, y)])
        return float(n_correct) / X.shape[0]

    def score2(self, X, y):
        proba = self.predict_proba(X)
        y_ind = [self.classes_rev[c] for c in y]
        return np.mean([row[ind] for row, ind in zip(proba, y_ind)])

    def cross_validation(self, X, y, alphas, k=5):
        n_samples = X.shape[0]
        assert(len(y) == n_samples)
        indices = range(n_samples)
        np.random.shuffle(indices)
        split = [sorted(indices[start::k]) for start in range(k)]
        split_X = [X[ind_list] for ind_list in split]
        split_y = [np.array(y)[ind_list] for ind_list in split]
        scores = dict()
        for alpha in alphas:
            scores[alpha] = []
            for leave_out in range(k):
                X_train = sp.sparse.vstack(tuple(split_X[i] for i in range(k) if i != leave_out))
                y_train = np.concatenate(tuple(split_y[i] for i in range(k) if i != leave_out))
                self.fit(X_train, y_train, alpha)
                scores[alpha].append(self.score(split_X[leave_out], split_y[leave_out]))
        return scores

    def representative_words(self, n_words=10):
        logwordprobs_avg = np.mean(self.logwordprobs, axis=0)
        logwordprobs_normalized = self.logwordprobs - logwordprobs_avg
        indices = [sorted(range(len(lst)), key = lambda ind : -lst[ind])[0:n_words] for lst in logwordprobs_normalized]
        if self.vocab_rev is None:
            return indices
        else:
            return [[self.vocab_rev[ind] for ind in lst] for lst in indices]

    def representative_errors(self, X, y, n_examples=10):
        proba = self.predict_proba(X)
        posterior = np.array([lst[self.classes_rev[c]] for lst, c in zip(proba, y)])
        posterior_split = [posterior[y == self.classes_rev[c]] for c in self.classes]
        extreme_errors = [sorted(range(len(lst)), key = lambda ind : lst[ind])[0:n_examples] for lst in posterior_split]
        class_indices = [np.array(range(X.shape[0]))[y == self.classes_rev[c]] for c in self.classes]
        #print class_indices[0][0:20]
        #print class_indices[1][0:20]
        extreme_errors = [[ind_lst[err_ind] for err_ind in err_lst] for err_lst, ind_lst in zip(extreme_errors, class_indices)]
        return extreme_errors

class NotNaiveBayes:
    def fit(self, X, y, n_preceding=1, alpha=1., vocab=None):
        self.X = X
        self.y = np.array(y)
        self.n_preceding = n_preceding
        self.alpha = alpha
        self.vocab = vocab
        self.vocab_rev = {v:k for k,v in vocab.items()} if vocab is not None else None
        self.vocab_size = len(vocab) if vocab is not None else max(map(max, X))+1
        self.classes = list(set(self.y))
        self.classes_rev = {v:k for k,v in enumerate(self.classes)}
        self.nclasses = len(self.classes)

        self.classcounts = np.array([float((self.y==c).sum()) for c in self.classes])
        self.classprobs = self.classcounts / self.classcounts.sum()
        self.logclassprobs = np.log(self.classprobs)

        self.transition_mats = []
        for c in self.classes:
            cb_mat = sp.sparse.lil_matrix((self.vocab_size**self.n_preceding+1, self.vocab_size+1))
            for lst in [x for x, y in zip(X, y) if y == c]:
                lst_new = [cb_mat.shape[0]-1] + lst + [cb_mat.shape[1]-1]
                seq_lst = zip(*(tuple(lst_new[i:] for i in range(n_preceding+1))))
                for tup in seq_lst:
                    cb_mat[self.compute_index(tup)] += 1
            cb_mat = sp.sparse.lil_matrix(cb_mat / cb_mat.sum(axis=1))
            assert((cb_mat.sum(axis=1) == 1)[0].all())
            self.transition_mats.append(cb_mat.copy())

    def compute_index(self, tup):
        return sum([elt*(self.vocab_size**exponent) for elt, exponent in zip(tup[:-1], reversed(range(len(tup)-1)))]), tup[-1]