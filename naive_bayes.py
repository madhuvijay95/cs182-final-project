import numpy as np
import scipy as sp

class NaiveBayes:
    def fit(self, X, y, alpha=1.):
        self.X = np.array(X)
        self.y = np.array(y)
        self.alpha = alpha
        self.classes = list(set(self.y))
        self.classes_rev = {v:k for k,v in enumerate(self.classes)}
        self.nclasses = len(self.classes)

        self.classcounts = np.array([float((self.y==c).sum()) for c in self.classes])
        self.classprobs = self.classcounts / self.classcounts.sum()
        self.logclassprobs = np.log(self.classprobs)

        self.wordcounts = np.array([map(float, np.array((X[y==c].todense()).sum(axis=0))[0]) for c in self.classes])
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
