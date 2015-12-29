# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 10:48:11 2015

@author: Guillaume
"""

import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds

csv_input = '../input/dm_mec_ng_bo.csv'

#==============================================================================
# Apply dimensionnality reduction
#==============================================================================

dataframe = pd.read_csv(csv_input)

dataframe = dataframe[:50000]

listIndividu = pd.unique(dataframe['INDIV_ID'].values)
nbindiv = len(listIndividu)

listOffre = pd.unique(dataframe['JOBOFFER_ID'].values)
nbOffre = len(listOffre)

rows = list(dataframe['INDIV_ID'])
cols = list(dataframe['JOBOFFER_ID'])
vals = list(dataframe['SCORE'])

shape = (nbindiv, nbOffre)

m = coo_matrix((vals, (rows, cols)), shape=shape)

u,s,vt = svds(m,k=15)