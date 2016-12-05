import numpy as np
from scipy.sparse import linalg
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
import time

# adapted from krzysztof sopyla's implementation at https://github.com/ksopyla/primal_svm
class SupportVectorMachine:
    
    # fits the kNN model using gradient descent and a linear kernel
    # input:    (1) matrix X of tf-idf frequencies (e.g. using the output of sklearn's TfidfVectorizer)
    #           (2) list y of class assignments (binary and signed, i.e. -1 and 1)
    # output:   (1) list of predictions for class assignments (based on sign)
    
    def __init__(self, lmbda=1., n_features=50):
        self.lmbda = lmbda
        self.n_features = n_features
    
    def fit_score(self, traindf, testdf, full_report=True):
        
        start_time = time.time()
        
        # extract y vector from data
        y_train = np.array(traindf.clickbait)
        y_test = np.array(testdf.clickbait)
        
        # vectorize x data
        vectorizer = TfidfVectorizer(max_features=self.n_features, stop_words='english', use_idf=True)
        x_train = np.asarray(vectorizer.fit_transform(list(traindf.article_title)).todense())
        x_test = np.asarray(vectorizer.transform(list(testdf.article_title)).todense())
        
        # initialize vector of zeros normal to the hyperplane
        w = np.zeros(x_train.shape[1]+1)
        
        # loss function values
        loss = np.ones(x_train.shape[0])

        # generate support vectors (training points lying on one of the hyperplanes)
        sv = np.where(loss > 0)[0]
        
        # compute linear operator in replacement of full hessian
        mv = lambda v: self.matrix_multiply(x_train, v, sv)
        l_op = linalg.LinearOperator((x_train.shape[1]+1, x_train.shape[1]+1), matvec = mv)
        
        # perform conjugate gradient ascent using loss function and linear operator
        self.gradient_descent(x_train, y_train, w, loss, l_op)
        
        # take dot products between vectors in test data 
        scores = x_test.dot(w[0:-1]) + w[-1]
        
        # generate prediction based on signs of scores (+ = clickbait, - = not)
        predictions = np.sign(scores)
        
        # summarize model performance
        self.accuracy = accuracy_score(y_test, predictions)
        print "took %s seconds" % (time.time() - start_time)
        print "overall accuracy is " + str(self.accuracy) 
        if full_report:
            print "full report:"
            print 
            print classification_report(y_test, predictions, target_names = ['clickbait','non-clickbait'])

    # perform conjugate gradient ascent
    def gradient_descent(self, x, y, w, l, hv):
        
        # fix number of steps
        for i in range(100):
            max_l = np.fmax(0, l)
            
            # compute objective function value
            obj = np.sum(max_l**2)/2 + self.lmbda*w.dot(w)/2
            
            # compute gradient
            grad = self.lmbda*w - np.append([np.dot(max_l*y,x)], [np.sum(max_l*y)])
            
            # perform line search for optimal w (normal to hyperplane)
            sv = np.where(l > 0)[0]
            
            # vector where optimal solution is located
            vec, info = linalg.minres(hv, -grad)
            x_d = x.dot(vec[0:-1]) + vec[-1]
            w_d = self.lmbda*w[0:-1].dot(vec[0:-1])
            d_d = self.lmbda*vec[0:-1].dot(vec[0:-1])
            
            t = 0
            l_0 = l
            for i in range(1000):
                l_0 = l-t*(y*x_d)
                sv = np.where(l_0 > 0)[0]

                g = w_d+t*d_d-(l_0[sv]*y[sv]).dot(x_d[sv])
                h = d_d + x_d[sv].dot(x_d[sv])
                t -= g/h
            
            w += t*vec
            if -vec.dot(grad) < 0.0000001 * obj:
                break
    
    # multiply big matrix and vector without intermediary square matrix 
    def matrix_multiply(self, x, v, sv):
        y = self.lmbda * v
        y[-1] = 0
        
        x_0 = x.dot(v[0:-1]) + v[-1]
        x_1 = np.zeros(x_0.shape[0])
        x_1[sv] = x_0[sv]
        y = y + np.append(x_1.dot(x), x_1.sum())
        
        return y