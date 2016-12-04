import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from knn import kNearestNeighbors
import sys
import time

# import data
filename = 'augmented.csv'
df = pd.read_csv(filename)

# lower case
df.article_title = [x.lower() for x in df.article_title]

# randomly sample fraction of data for testing
# df = df.sample(frac=0.2)

print '====================== hand kNN RESULTS ======================'

knn = kNearestNeighbors()
knn.fit_score(df, k=3, n_features=50)

# cross-validation
# knn.cv(df, k_vals=[1,3,5,7,9])
# knn.cv(df, n_vals=[10,50,100,250,500])

sys.stdout.flush()