#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 16:40:59 2022

@author: GR
"""

import pickle
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.model_selection import GridSearchCV

with open("Data/xtrain", 'rb') as reader:
    X_train = pickle.load(reader)

with open("Data/ytrain", 'rb') as reader:
    y_train = pickle.load(reader)

with open("Data/xtest", 'rb') as reader:
    X_test = pickle.load(reader)

with open("Data/ytest", 'rb') as reader:
    y_test = pickle.load(reader)
del reader

# %%

mod = LinearRegression()
my_fit = mod.fit(X_train, y_train)
mod.score(X_train, y_train)

print("score train :", my_fit.score(X_train, y_train))
print("score test :", my_fit.score(X_test, y_test))

plt.scatter(y_test, my_fit.predict(X_test), alpha=0.25)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], "r-")

plt.xlabel('Année de plantation')
plt.ylabel('Année de plantation prédite')
plt.show()

# %% Elastic net

mod = ElasticNet()
parm = {'l1_ratio': [0.1, 0.25, 0.5, 0.75, 0.9],
        'alpha': [0.5, 1, 10, 30, 100]}
grid = GridSearchCV(mod, parm, verbose=2)

search = grid.fit(X_train, y_train)
search.best_params_
search.scorer_
search.best_score_

# %%
mod = ElasticNet(alpha=0.5, l1_ratio=0.9)
mod.fit(X_train, y_train)
print("score train :", mod.score(X_train, y_train))
print("score test :", mod.score(X_test, y_test))

# %% Save

with open('Data/linear', 'wb') as writer:
    pickle.dump(my_fit, writer)


with open('Data/elasticNet', 'wb') as writer:
    pickle.dump(mod, writer)
del writer
