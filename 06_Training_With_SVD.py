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

nbTestSet = 10000
listCoordinateTestSet = []

print "Creation of test set"
n = 0
while n < nbTestSet:
    rand = np.random.randint(0,nbmec)
    vals[rand] = 0
    if (rows[rand],cols[rand]) not in listCoordinateTestSet:
        n += 1
        listCoordinateTestSet.append((rows[rand],cols[rand]))

print "Creation of test set OK"
print "Creation of train set"
listCoordinateTrainSet = []
for i in range(nbmec):
    if ((rows[i],cols[i])) not in listCoordinateTestSet:
        listCoordinateTrainSet.append((rows[i],cols[i]))
print "Creation of train set OK"
shape = (nbIndiv, nbOffre)
m = coo_matrix((vals, (rows, cols)), shape=shape)
m = m.tocsr()

# How many dimensions?
k = 800
print "Computation of SVD"
# Initialize the matrix using a singular value decomposition
u,s,vt = svds(m,k = k)
print "Computation of SVD OK"
# We're now looking for P and Q such as R = P.Qt
P = (u.dot(np.diag(s)))
Q = vt
print "Shape of P: %s" % str(P.shape)
print "Shape of Q: %s" % str(Q.shape)

#R = P.dot(Q)

'''for (indiv,offre) in listCoordinateTestSet:
    print P[indiv,:].dot(Q[:,offre])'''
    #print R[indiv,offre]

#seuilSuccess = 0.05
nbSuccessTestSet = 0
nbSuccessTrainSet = 0
nbPositifHorsSet = 0
i = 0
j = 0
    
listTrainSetResult = []
print "Computation of success in train set"
nbTrainSet = nbmec - nbTestSet
print "nbTrainSet = %i" % nbTrainSet
for (indiv,offre) in listCoordinateTrainSet:
    prediction = P[indiv,:].dot(Q[:,offre])
    listTrainSetResult.append(prediction)
    '''if prediction > seuilSuccess:
        nbSuccessTrainSet += 1'''       
q = 10
seuilSuccess = float(np.percentile(listTrainSetResult,q))
print "Seuil de succes positionne a: %1.5f" % seuilSuccess
for prediction in listTrainSetResult:
    if prediction > seuilSuccess:
        nbSuccessTrainSet += 1
print "Computation of success in train set OK"
print "nbSuccessTrainSet = %i" % nbSuccessTrainSet
print "Taux de success Train Set: %1.1f" % (100*nbSuccessTrainSet/float(nbTrainSet))
    
    
print "Computation of success in test set"
print "nbTestSet = %i" % nbTestSet
for (indiv,offre) in listCoordinateTestSet:
    prediction = P[indiv,:].dot(Q[:,offre])
    if prediction > seuilSuccess:
        nbSuccessTestSet += 1
print "Computation of success in test set OK"
print "nbSuccessTestSet = %i" % nbSuccessTestSet
print "Taux de success Test Set: %1.1f" % (100*nbSuccessTestSet/float(nbTestSet))


print "Computation of success in random data"
nbData = 50000
ntime = 0
for i in range(nbData):
    indiv = np.random.randint(0,nbIndiv)
    offre = np.random.randint(0,nbOffre)
    if (indiv,offre) not in listCoordinateTestSet: 
        if (indiv,offre) not in listCoordinateTrainSet:
            ntime += 1
            prediction = P[indiv,:].dot(Q[:,offre])
            if prediction > seuilSuccess:
                nbPositifHorsSet += 1                
print "Taux positif Hors Set: %1.1f" % (100*nbPositifHorsSet/float(ntime))
print "nb Hors set = %i" % ntime
print "nbPositifHorsSet = %i" % nbPositifHorsSet
   
'''for indiv in range(0,nbIndiv):
    for offre in range(0,nbOffre):
        i += 1
        if i%10000 == 0:
            print "Number i up to %i" % i
        reality = m[indiv,offre]
        if (indiv,offre) in listCoordinateTestSet:
            prediction = P[indiv,:].dot(Q[:,offre])
            if prediction > seuilSuccess:
                nbSuccessTestSet += 1
        else:
            if reality == 1:
                prediction = P[indiv,:].dot(Q[:,offre])
                if prediction > seuilSuccess:
                    nbSuccessTrainSet += 1
            elif j < 10000:
                j += 1
                if j%1000 == 0:
                    print "################### Number j up to %i" % j
                prediction = P[indiv,:].dot(Q[:,offre])
                if prediction > seuilSuccess:
                    nbPositifHorsSet += 1'''
        





'''print "#####################"
print P[rows[0],:].dot(Q[:,cols[0]])
print P[rows[1],:].dot(Q[:,cols[1]])
print P[rows[2],:].dot(Q[:,cols[2]])
print P[rows[3],:].dot(Q[:,cols[3]])
print P[rows[4],:].dot(Q[:,cols[4]])'''





# Now let's look at a gradient descent to find P.Qt such as we always predict the
# appropriate value of the utility matrix when known


