class NaiveBayes:
    def train(self, X, y, alpha):
        self.X = np.array(X)
        self.y = np.array(y)
        self.classes = list(set(self.y))
        self.nclasses = len(self.classes)

        self.classcounts = [float((y==c).sum()) for c in self.classes]
        self.classprobs = (self.classcounts.T / self.classcounts.sum(axis==1)).T
        self.logclassprobs = np.log(self.classprobs)

        self.wordcounts = np.array([float((X[y==c] + alpha).sum(axis=0)) for c in self.classes])
        self.wordprobs = (self.wordcounts.T / self.wordcounts.sum(axis=1)).T
        self.logwordprobs = np.log(self.wordprobs)

    def predict_log_proba(self, X):
        log_proba = np.dot(X, self.logwordprobs.T)
        log_proba += self.logclassprobs
        log_proba -= np.log(np.exp(log_proba))
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
        return float(n_correct) / X.shape[1]

    def cross_validation(self, X, y, alphas, k=5):
        n_samples = X.shape[1]
        assert(len(y) == n_samples)
        indices = range(n_samples)
        np.random.shuffle(indices)
        split = [sorted(indices[start::k]) for start in range(k)]
        split_X = [np.array(X)[ind_list] for ind_list in split]
        split_y = [np.array(y)[ind_list] for ind_list in split]
        scores = dict()
        for alpha in alphas:
            scores[alpha] = []
            for leave_out in range(k):
                X_train = np.hstack(tuple(split_X[i] for i in range(k) if i != leave_out))
                y_train = np.concatenate(tuple(split_y[i] for i in range(k) if i != leave_out))
                self.train(X_train, y_train, alpha)
                scores[alpha].append(self.score(split_X[leave_out], split_y[leave_out]))
        return scores
