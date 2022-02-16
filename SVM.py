#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 16:12:29 2022

@author: GR
"""

from sklearn.svm import SVR
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import uniform

with open("Data/xtrain", 'rb') as reader:
    X_train = pickle.load(reader)

with open("Data/ytrain", 'rb') as reader:
    y_train = pickle.load(reader)

with open("Data/xtest", 'rb') as reader:
    X_test = pickle.load(reader)

with open("Data/ytest", 'rb') as reader:
    y_test = pickle.load(reader)
del reader

# %% Select hyperparameters
svm = SVR(max_iter=1000)
param_grid = [
    {'C': [1, 50, 1000], 'epsilon': [0.1, 0.5, 1], 'kernel': ['rbf']}]
mysearch = GridSearchCV(svm, param_grid, verbose=3, cv=2)
mysearch.fit(X_train, y_train)
print(mysearch.best_params_)

# %% RandomizedSearch
svm = SVR(max_iter=1000)
distributions = dict(C=uniform(loc=30, scale=30))
search = RandomizedSearchCV(svm, param_distributions=distributions)
search.fit(X_train, y_train)
print(search.best_params_)

# %% It's time to SVM
svm = SVR(max_iter=-1, C=54, verbose=True)
my_fit = svm.fit(X_train, y_train)

# %% Score and graph

print("score train :", my_fit.score(X_train, y_train))
print("score test :", my_fit.score(X_test, y_test))

plt.scatter(y_test, my_fit.predict(X_test), alpha=0.25)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], "r-")

plt.xlabel('Année de plantation')
plt.ylabel('Année de plantation prédite')
plt.show()

# %% Save model

with open('Data/myfit', 'wb') as writer:
    pickle.dump(my_fit, writer)
del writer
