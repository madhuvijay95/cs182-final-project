import numpy as np
from operator import itemgetter
from collections import Counter
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import cross_validation
import time

class kNearestNeighbors:
    # fits the kNN model
    # input: 	(1) matrix X of tf-idf frequencies (e.g. using the output of sklearn's TfidfVectorizer)
    #		 	(2) list y of class assignments
    # output: 	(1) list of predictions for class assignments

	def __init__(self, k=9, n_features=100000):
		self.k = k
		self.n_features = n_features

	# generate predictions for data (in pandas dataframe format)
	def fit_score(self, traindf, testdf, k=9, n=100000, full_report=True, progress=True):

		self.k = k
		self.n_features = n

		start_time = time.time() 

		# extract y vector from data
		y_train = np.array(traindf.clickbait)
		y_test = np.array(testdf.clickbait)

		# vectorize x data
		vectorizer = TfidfVectorizer(max_features=self.n_features, stop_words='english', use_idf=True)
		x_train = np.asarray(vectorizer.fit_transform(list(traindf.article_title)).todense())
		x_test = np.asarray(vectorizer.transform(list(testdf.article_title)).todense())
		
		# split data into test and train sets
		# x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=0.3, random_state=1)

		# recombine (x, y) for each set
		train = np.array(zip(x_train, y_train))
		test = np.array(zip(x_test, y_test))

		# turn train and test word counts into matrices
		train_mat = np.array([row[0] for row in train])
		test_mat = np.array([row[0] for row in test])


		# generate a matrix containing the Euclidean distance between each row of the train data and each row of the
		# test data, using the fact that ||x-y||^2 = ||x||^2 + ||y||^2 - 2x^T*y for any vectors x and y
		dists = np.dot(np.ones((test_mat.shape[0], 1)), np.array([np.linalg.norm(train_mat, axis=1)**2]))\
				+ np.dot(np.array([np.linalg.norm(test_mat, axis=1)**2]).T, np.ones((1, train_mat.shape[0])))\
				- 2 * np.dot(test_mat, train_mat.T)

		# find the k closest neighbors from the train data for each test data point
		neighbors_all = [sorted(enumerate(row), key = lambda tup : tup[1])[0:self.k] for row in dists]
		neighbors_all = [[(tup[0], y_train[tup[0]]) for tup in row] for row in neighbors_all]
		
		# use majority_vote to generate predictions
		predictions = [self.majority_vote(neighbors) for neighbors in neighbors_all]

		# summarize model performance
		self.accuracy = accuracy_score(y_test, predictions)
		print "took %s seconds" % (time.time() - start_time)
		print "overall accuracy is " + str(self.accuracy)
		if full_report:
			print "full report:"
			print
			print classification_report(y_test, predictions, target_names = ['clickbait','non-clickbait'])

	# calculate euclidean distance between two data points x_1 and x_2
	def get_distance(self, x_1, x_2):
	    # return square root of sum of squared distances
	    sq_distances = [(a - b)**2 for (a, b) in zip(x_1, x_2)]
	    eu_distance = (sum(sq_distances))**0.5
	    return eu_distance

	# get k closest neighbors in training data to test
	def get_neighbors(self, train_data, test_obs, k):
		# calculate distances between test data point and each train obs
		distances = [(train_obs, self.get_distance(test_obs, train_obs[0])) for train_obs in train_data]

		# return k closest neighbors by distance
		sorted_train_obs = [tuple[0] for tuple in sorted(distances, key=itemgetter(1))]
		return sorted_train_obs[:k]

	# vote on test observation class assignment with neighbor classes
	def majority_vote(self, neighbors):
		# get classes of neighbors
		classes = [neighbor[1] for neighbor in neighbors]
		
		# return majority class 
		return Counter(classes).most_common()[0][0]

	# score model accuracy
	def score(y_test, preds):
		return accuracy_score(y_test, preds)

	# cross validate across values of k or n (tfidf n_features param)
	def cv(self, traindf, testdf, k_vals=None, n_vals=None):
		# store accuracies to find best value
		accuracies = {}
		
		if k_vals is not None:
			for k in k_vals:
				print '========== k = %s ==========' % k
				self.fit_score(traindf=traindf, testdf=testdf, k=k, n=100000, full_report=False, progress=False)
				accuracies[k] = self.accuracy
			print "best value of k is", max(accuracies.iteritems(), key=itemgetter(1))
		
		if n_vals is not None:
			for n in n_vals:
				print '========== n = %s ==========' % n
				self.fit_score(traindf=traindf, testdf=testdf, k=9, n=n, full_report=False, progress=False)
				accuracies[n] = self.accuracy
			print "best value of n is", max(accuracies.iteritems(), key=itemgetter(1))