# split dataframe into test and train
# for cross-algorithm testing

import pandas as pd
import numpy as np

# import data
filename = 'augmented.csv'
df = pd.read_csv(filename)
del df['Unnamed: 0']

# 70 / 30 split for train / test
msk = np.random.rand(len(df)) < 0.7
train = df[msk]
test = df[~msk]

train.to_csv("train.csv", index=False)
test.to_csv("test.csv", index=False)