#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 15:01:34 2022

@author: GR
"""
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd
import pickle

wd = "/home/dataplus/Grivollier/Machine learning/ML1 - introduction au ML/"
df = pd.read_pickle('Data/CleanDatawithANNEE')


# %% Split
y = df.pop('ANNEEDEPLANTATION')
X_train, X_test, y_train, y_test = train_test_split(df, y, random_state=91)
# %% create Pipelines

# Cat : sous categorie et sous categories desc, code parent et code parents desc,
# genre bota, espece, variete, remarques, collectivité

# ord : developpement

# num : elem point id, adr secteur, annee de plantation (VD) ,lon lat

col_cat = ['SOUS_CATEGORIE', 'SOUS_CATEGORIE_DESC',
           'CODE_PARENT_DESC', 'GENRE_BOTA', 'ESPECE', 'VARIETE',
           'COLLECTIVITE']

col_ord = ['STADEDEDEVELOPPEMENT']

col_num = ['ELEM_POINT_ID', 'ADR_SECTEUR', 'lat', 'lon']
colvalues = {}
for i in col_cat:
    colvalues[i] = pd.unique(df[i])


cat_pipeline = Pipeline([
    ('hot', OneHotEncoder(handle_unknown='ignore')),
])

ord_pipeline = Pipeline([
    ('imp,', SimpleImputer(strategy='constant', fill_value='None')),
    ('ord', OrdinalEncoder()),
    ('scale', StandardScaler())
])

num_pipeline = Pipeline([
    ('imp', SimpleImputer(strategy='mean')),
    ('scale', StandardScaler())
])

col_ord = ['STADEDEDEVELOPPEMENT']

col_num = ['ELEM_POINT_ID', 'ADR_SECTEUR', 'lat', 'lon']


full_pipeline = ColumnTransformer([
    ('cat', cat_pipeline, col_cat),
    ('ord', ord_pipeline, col_ord),
    ('num', num_pipeline, col_num)])


# %% apply pipeline

X_train = full_pipeline.fit_transform(X_train)
X_test = full_pipeline.transform(X_test)

# %% Save

with open('Data/xtrain', 'wb') as writer:
    pickle.dump(X_train, writer)

with open('Data/ytrain', 'wb') as writer:
    pickle.dump(y_train, writer)

with open('Data/xtest', 'wb') as writer:
    pickle.dump(X_test, writer)

with open('Data/ytest', 'wb') as writer:
    pickle.dump(y_test, writer)

with open('Data/transformer', 'wb') as writer:
    pickle.dump(full_pipeline, writer)
del writer
