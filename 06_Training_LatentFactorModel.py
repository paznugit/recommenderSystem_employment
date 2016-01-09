# -*- coding: utf-8 -*-
"""
Created on Tue Jan 05 13:40:03 2016

@author: IGPL3460
"""


import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds
from scipy import linalg as LA
import numpy as np

# Extract the utility matrix (link between individual and job offer)
csv_input = '../input/dm_mec_21_ng_bo.csv'
df_utility = pd.read_csv(csv_input)

# Get the shape of the utility matrix
nbIndiv = len(pd.unique(df_utility['INDIV_ID'].values))
nbOffre = len(pd.unique(df_utility['JOBOFFER_ID'].values))

# TODO: Remove some line of utility dataframe to get a test data

print "number of mec: %i" % df_utility.index
print "number of individuals: %i" % nbIndiv
print "number of job offers: %i" % nbOffre

# Instanciate a sparse matrix
rows = list(df_utility['INDIV_ID'])
cols = list(df_utility['JOBOFFER_ID'])
vals = [float(x) for x in list(df_utility['SCORE'])]

shape = (nbIndiv, nbOffre)
m = coo_matrix((vals, (rows, cols)), shape=shape)

# How many latent factors?
k = 20
# Initialize the matrix using a singular value decomposition
u,s,vt = svds(m,k = 20)

# We're now looking for P and Q such as R = P.Qt
P = (u.dot(np.diag(s)))
Q = vt
print "Shape of P" % shape(P)
print "Shape of Q" % shape(Q)

# Now let's look at a gradient descent to find P.Qt such as we always predict the
# appropriate value of the utility matrix when known


