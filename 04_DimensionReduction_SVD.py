# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 10:48:11 2015

@author: Guillaume
"""

import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds
from scipy import linalg as LA
import numpy as np

csv_input = '../input/dm_mec_ng_bo.csv'

#==============================================================================
# Apply dimensionnality reduction
#==============================================================================

dataframe = pd.read_csv(csv_input)

'''nbIndiv = 80000
nbOffre = 100000
listIndiv = range(nbIndiv)
listOffre = range(nbOffre)

dataframe = dataframe[dataframe['INDIV_ID'].isin(listIndiv)]
dataframe = dataframe[dataframe['JOBOFFER_ID'].isin(listOffre)]'''

'''listIndividu = pd.unique(dataframe['INDIV_ID'].values)
nbIndiv = len(listIndividu)

listOffre = pd.unique(dataframe['JOBOFFER_ID'].values)
nbOffre = len(listOffre)'''

nbIndiv = len(pd.unique(dataframe['INDIV_ID'].values))
nbOffre = len(pd.unique(dataframe['JOBOFFER_ID'].values))
print nbIndiv
print nbOffre

#print dataframe.loc[dataframe['JOBOFFER_ID'] > 948915]

rows = list(dataframe['INDIV_ID'])
cols = list(dataframe['JOBOFFER_ID'])
vals = [float(x) for x in list(dataframe['SCORE'])]

shape = (nbIndiv, nbOffre)

m = coo_matrix((vals, (rows, cols)), shape=shape)

#u,s,vt = svds(m,k=6)

for k in range(300):
    u,s,vt = svds(m,k=k)
    print "k=%i" %k
    print s.shape

#a = (u.dot(np.diag(s))).dot(vt)


#print LA.norm(a-m,ord ='fro')
'''print u.shape
print s.shape
print vt.shape
print np.diag(s)'''

'''print m
print u
print s
print vt'''