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
nbmec = len(df_utility.index)

print "number of mec: %i" % nbmec
print "number of individuals: %i" % nbIndiv
print "number of job offers: %i" % nbOffre

# Instanciate a sparse matrix
rows = list(df_utility['INDIV_ID'])
cols = list(df_utility['JOBOFFER_ID'])
vals = [float(x) for x in list(df_utility['SCORE'])]

# TODO: Remove some line of utility dataframe to get a test data
nbTestSet = 50000
listCoordinateTestSet = []

for i in range(nbTestSet):
    rand = np.random.randint(0,nbmec)
    vals[rand] = 0
    listCoordinateTestSet.append((rows[rand],cols[rand]))

shape = (nbIndiv, nbOffre)
m = coo_matrix((vals, (rows, cols)), shape=shape)
m = m.tocsr()
 
print m[rows[0],cols[0]]

# How many latent factors?
k = 800
# Initialize the matrix using a singular value decomposition
u,s,vt = svds(m,k = k)

# We're now looking for P and Q such as R = P.Qt
P = (u.dot(np.diag(s)))
Q = vt
print type(P)
print "Shape of P: %s" % str(P.shape)
print "Shape of Q: %s" % str(Q.shape)

#R = P.dot(Q)

'''for (indiv,offre) in listCoordinateTestSet:
    print P[indiv,:].dot(Q[:,offre])'''
    #print R[indiv,offre]

seuilSuccess = 0.05
nbSuccessTestSet = 0
nbSuccessTrainSet = 0
nbPositifHorsSet = 0
i = 0
for indiv in range(0,nbIndiv):
    for offre in range(0,nbOffre):
        i += 1
        if i%10000 == 0:
            print i
        prediction = P[indiv,:].dot(Q[:,offre])
        reality = m[indiv,offre]
        if (indiv,offre) in listCoordinateTestSet:
            if prediction > seuilSuccess:
                nbSuccessTestSet += 1
        else:
            if reality == 1:
                if prediction > seuilSuccess:
                    nbSuccessTrainSet += 1
            else:
                if prediction > seuilSuccess:
                    nbPositifHorsSet += 1
        
nbTrainSet = nbmec - nbTestSet
print "Taux de success Test Set: %1.1f" % nbSuccessTestSet/float(nbTestSet)
print "Taux de success Train Set: %1.1f" % nbSuccessTrainSet/float(nbTrainSet)
print "Nb positif Hors Set: %i" % nbPositifHorsSet

'''print "#####################"
print P[rows[0],:].dot(Q[:,cols[0]])
print P[rows[1],:].dot(Q[:,cols[1]])
print P[rows[2],:].dot(Q[:,cols[2]])
print P[rows[3],:].dot(Q[:,cols[3]])
print P[rows[4],:].dot(Q[:,cols[4]])'''





# Now let's look at a gradient descent to find P.Qt such as we always predict the
# appropriate value of the utility matrix when known


