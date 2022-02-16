# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 14:27:57 2022

@author: rivol
"""
import scipy as sp
import numpy as np
import pickle5 as pickle
import pandas as pd
import matplotlib.pyplot as plt

with open("Data/xtrain", 'rb') as reader:
    X_train = pickle.load(reader)

with open("Data/ytrain", 'rb') as reader:
    y_train = pickle.load(reader)

with open("Data/xtest", 'rb') as reader:
    X_test = pickle.load(reader)

with open("Data/ytest", 'rb') as reader:
    y_test = pickle.load(reader)

with open("Data/myfit", 'rb') as reader:
    SVR = pickle.load(reader)

with open("Data/myforest", 'rb') as reader:
    RF = pickle.load(reader)

with open("Data/linear", 'rb') as reader:
    LM = pickle.load(reader)

with open("Data/elasticNet", 'rb') as reader:
    EN = pickle.load(reader)

with open("Data/transformer", 'rb') as reader:
    pipeline = pickle.load(reader)

with open("Data/CleanDataWithoutANNEE", 'rb') as reader:
    data_to_predict = pickle.load(reader)

del reader


# %% Training data

X = sp.sparse.vstack((X_train, X_test))
y = np.hstack((y_train, y_test))

# %% And retrain all models

SVR.fit(X, y)
RF.fit(X, y)
LM.fit(X, y)
EN.fit(X, y)

# %% print scores

print("SVM score :", SVR.score(X, y))
print("RF score :", RF.score(X, y))
print("LM score :", LM.score(X, y))
print("EN score :", EN.score(X, y))

# SVM score : 0.9505567500781832
# RF score : 0.929523047434164
# LM score : 0.8489963223992394
# EN score : 0.33145650905568647
# %% Predictions

data = pipeline.transform(data_to_predict)

SVR_pred = pd.DataFrame(SVR.predict(data))
RF_pred = pd.DataFrame(RF.predict(data))
LM_pred = pd.DataFrame(LM.predict(data))
EN_pred = pd.DataFrame(EN.predict(data))

# %% Shows preds
pred = pd.concat([SVR_pred, RF_pred, LM_pred, EN_pred], axis=1)
pred.columns = ['SVR', 'RF', 'LM', 'EN']
pred.sort_values(by='SVR', inplace=True)
plt.scatter(range(len(pred)), pred.SVR, label='SVR', c='r', alpha=0.2)
plt.scatter(range(len(pred)), pred.RF, label='RF', c='g', alpha=0.2)
plt.scatter(range(len(pred)), pred.LM, label='LM', c='k', alpha=0.2)
plt.scatter(range(len(pred)), pred.EN, label='EN', c='b', alpha=0.2)
plt.legend()
plt.show()

# %% Save models

dict = {"SVR": SVR, 'RF': RF, "LM": LM, 'EN': EN}

with open('Data/Allmodels train on all data', 'wb') as writer:
    pickle.dump(dict, writer)
del writer
