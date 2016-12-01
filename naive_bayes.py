import numpy as np
import scipy as sp
from scipy import sparse

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
    def convert(self, dataset, countvectorizer):
        analyzer = countvectorizer.build_analyzer()
        return [[countvectorizer.vocabulary_[word] for word in analyzer(title) if word in countvectorizer.vocabulary_] for title in dataset] # TODO this is weird because it just totally ignores words that aren't in the CountVectorizer vocab

    def fit(self, X, y, n_preceding=1, alpha=1., vocab=None):
        self.X = X
        self.y = np.array(y)
        self.n_preceding = n_preceding
        self.alpha = alpha
        self.vocab = vocab
        self.vocab_rev = {v:k for k,v in vocab.items()} if vocab is not None else None
        self.vocab_size = len(vocab) if vocab is not None else max([max(lst) for lst in X if len(lst) > 0])+1
        self.classes = list(set(self.y))
        self.classes_rev = {v:k for k,v in enumerate(self.classes)}
        self.nclasses = len(self.classes)

        self.classcounts = np.array([float((self.y==c).sum()) for c in self.classes])
        self.classprobs = self.classcounts / self.classcounts.sum()
        self.logclassprobs = np.log(self.classprobs)

        self.count_mats = []
        for c in self.classes:
            cb_mat = sp.sparse.lil_matrix((self.vocab_size**self.n_preceding+1, self.vocab_size+1))
            for lst in [x for x, k in zip(X, y) if k == c]:
                lst_new = [cb_mat.shape[0]-1] + lst + [cb_mat.shape[1]-1]
                seq_lst = zip(*(tuple(lst_new[i:] for i in range(self.n_preceding+1))))
                for tup in seq_lst:
                    cb_mat[self.compute_index(tup)] += 1
            self.count_mats.append(cb_mat.copy())

    def compute_index(self, tup):
        return sum([elt*(self.vocab_size**exponent) for elt, exponent in zip(tup[:-1], reversed(range(len(tup)-1)))]), tup[-1]

    def predict_log_proba(self, X):
        X_modified = [[self.vocab_size**self.n_preceding] + lst + [self.vocab_size] for lst in X]
        X_modified = [zip(*(tuple(lst[i:] for i in range(self.n_preceding+1)))) for lst in X_modified]
        indices = [[self.compute_index(tup) for tup in lst] for lst in X_modified]
        log_proba = np.array([[sum([np.log(float(mat[tup[0], tup[1]] + self.alpha) / (mat[tup[0]].sum() + self.alpha * mat.shape[1])) for tup in lst]) for mat in self.count_mats] for lst in indices])
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
        return float(n_correct) / len(X)

    def generate(self, c):
        lst = [self.vocab_size**self.n_preceding]
        mat = self.count_mats[self.classes_rev[c]]
        while len(lst) == 1 or lst[-1] != self.vocab_size:
            dist = np.array(mat[lst[-1]].todense())[0]
            dist += self.alpha
            dist /= dist.sum()
            lst.append(np.random.choice(range(self.vocab_size+1), p=dist))
        if self.vocab is not None:
            return ' '.join([self.vocab_rev[ind] for ind in lst[1:-1]]).encode('utf-8')
        else:
            return lst

    def generate_mode(self, c):
        lst = [self.vocab_size**self.n_preceding]
        mat = self.count_mats[self.classes_rev[c]]
        while len(lst) == 1 or lst[-1] != self.vocab_size:
            dist = np.array(mat[lst[-1]].todense())[0]
            lst.append(np.argmax(dist))
        if self.vocab is not None:
            return ' '.join([self.vocab_rev[ind] for ind in lst[1:-1]]).encode('utf-8')
        else:
            return lst
