#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 10:00:14 2022

@author: GR
"""

from sklearn.preprocessing import OrdinalEncoder
import json
import pickle
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split  # NOQA
import numpy as np
import matplotlib.pyplot as plt

# %% Load data

data = pd.read_csv("Data/ARBRE.csv")

print(data.shape)
# Del useless columns
data.dropna(axis=1, how='all', inplace=True)
data.replace('nan', np.NaN, inplace=True)
# Del col with too many NA
data.dropna(axis=1, thresh=len(data)/10, inplace=True)
print(data.shape)
sns.heatmap(data.isnull(), cbar=False)
sns.heatmap(data.corr(), vmin=-1, vmax=1, cmap='bwr')
print(data.info())
print(data.describe())


# Récupérer les coordonnées sous forme de dictionnaires (un par row)
dic_coord = data['GeoJSON'].map(json.loads)

# Séparer longitude et latitude
lon = dic_coord.apply(lambda x: x['coordinates'][0])
lat = dic_coord.apply(lambda x: x['coordinates'][1])

# Ajouter les colonnes lon et lat à notre df, et enlever la colonne GeoJSON
data['lat'] = lat
data['lon'] = lon
data.drop('GeoJSON', axis=1, inplace=True)

# %% Filter

print(data.nunique())

data.drop(columns=(['CODE_PARENT', 'GENRE', 'GENRE_DESC', 'CATEGORIE',
                    'CATEGORIE_DESC', 'CODE', 'NOM', 'BIEN_REFERENCE',
                    'REMARQUES']), inplace=True)
print(data.shape)


# %% Write

with open("Data/arbre_filtre", 'wb') as writer:
    pickle.dump(data, writer)

del data, writer


# %% open with pickle

with open("Data/arbre_filtre", 'rb') as reader:
    data = pickle.load(reader)
del reader
print(data.head())

# %% separate data with / without anneedeplantation

data.replace({'nan': np.NaN})
datamiss = data[data.ANNEEDEPLANTATION.isnull()]
datahere = data.drop(datamiss.index)

datamiss.to_pickle('Data/CleanDataWithoutANNEE')
datahere.to_pickle('Data/CleanDatawithANNEE')
