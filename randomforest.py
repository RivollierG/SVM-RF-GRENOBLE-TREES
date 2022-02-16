# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 13:53:59 2022

@author: rivol

"""
from sklearn.ensemble import RandomForestRegressor
import pickle

with open("Data/xtrain", 'rb') as reader:
    X_train = pickle.load(reader)

with open("Data/ytrain", 'rb') as reader:
    y_train = pickle.load(reader)

with open("Data/xtest", 'rb') as reader:
    X_test = pickle.load(reader)

with open("Data/ytest", 'rb') as reader:
    y_test = pickle.load(reader)
del reader


# %% Declare model

forest = RandomForestRegressor(n_estimators=2500, min_samples_split=20,
                               max_features="sqrt", n_jobs=-1, verbose=3)
forest.fit(X_train, y_train)

# %% Evaluate

print(forest.score(X_train, y_train))

print(forest.score(X_test, y_test))

# Save model

with open('Data/myforest', 'wb') as writer:
    pickle.dump(forest, writer)
del writer
