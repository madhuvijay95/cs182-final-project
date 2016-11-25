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