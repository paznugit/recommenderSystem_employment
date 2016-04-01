# -*- coding: utf-8 -*-
"""
Created on Tue Jan 05 13:40:03 2016

@author: IGPL3460
"""

import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds
import numpy as np
from numpy import genfromtxt
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import LSHForest

# Parameter of this algorithm: The number of dimension used for SVD
k = 800
rerandomize = True

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

nbTestSet = 5000
listCoordinateTestSet = []

if rerandomize:
    # Creation of the test set
    print "Creation of test set"
    n = 0
    while n < nbTestSet:
        rand = np.random.randint(0,nbmec)
        vals[rand] = 0
        if (rows[rand],cols[rand]) not in listCoordinateTestSet:
            n += 1
            listCoordinateTestSet.append((rows[rand],cols[rand]))
    print "Creation of test set OK"
    
    # Creation of the train set
    print "Creation of train set"
    listCoordinateTrainSet = []
    for i in range(nbmec):
        if ((rows[i],cols[i])) not in listCoordinateTestSet:
            listCoordinateTrainSet.append((rows[i],cols[i]))
            if i > 15000:
                break
    print "Creation of train set OK"
else:
    # Do nothing
            
shape = (nbIndiv, nbOffre)
m = coo_matrix((vals, (rows, cols)), shape=shape)
m = m.tocsr()

# SVD computation
print "Computation of SVD"
# Initialize the matrix using a singular value decomposition
u,s,vt = svds(m,k = k)
print "Computation of SVD OK"
s = np.sqrt(s)
#print s
#print s[-2:]
# We're now looking for P and Q such as R = P.Qt
P = (u.dot(np.diag(s)))
Q = (np.diag(s)).dot(vt)
print "Shape of P: %s" % str(P.shape)
print "Shape of Q: %s" % str(Q.shape)
np.savetxt("../input/P_ini_SVD.csv", P, delimiter=",")
np.savetxt("../input/Q_ini_SVD.csv", Q, delimiter=",")
#P = genfromtxt("../input/P_ini.csv", delimiter=',')
#Q = genfromtxt("../input/Q_ini.csv", delimiter=',')
    
nbTrainSet = len(listCoordinateTrainSet) 

listeParameters = [800]
listPrecision = []
listRappel = []
listTestSetSuccess = []
for k in listeParameters:
    listeResult = []
    listeResult2 = []
    listesize = []
    nbSuccessTestSet = 0
    nbSuccessTrainSet = 0
    print "k = %i" % k
    
    # Computation of prediction for train set
    listTrainSetResult = []
    print "Computation of success in train set"
    print "nbTrainSet = %i" % nbTrainSet
    for (indiv,offre) in listCoordinateTrainSet:
        prediction = P[indiv,-k:].dot(Q[-k:,offre])
        listTrainSetResult.append(prediction)
    
    # Computation of seuilSuccess to have about 90% of success in train set
    q = 10
    seuilSuccess = float(np.percentile(listTrainSetResult,q))
    print "Seuil de succes positionne a: %1.5f" % seuilSuccess
    
    # Computation of success in train set
    for prediction in listTrainSetResult:
        if prediction > seuilSuccess:
            nbSuccessTrainSet += 1
    print "Computation of success in train set OK"
    print "nbSuccessTrainSet = %i" % nbSuccessTrainSet
    print "Taux de success Train Set: %1.1f" % (100*nbSuccessTrainSet/float(nbTrainSet))
    
    listeOffre = []
    # For each individual in test set, let's retrieve the list of job offer we would recommend
    listNbRecommend = []
    
    search = NearestNeighbors(n_neighbors=200, algorithm='auto').fit(P)
    #search = LSHForest(n_neighbors = 200,n_estimators  = 200, random_state=42).fit(P)
    for (indiv,offre) in listCoordinateTestSet:
        if offre in listeOffre:
            # Already done: We continue
            continue

        listeOffre.append(offre)
        
        distances, indices = search.kneighbors(Q[:,offre])
        if indiv in indices:
            nbSuccessTestSet += 1

        setIndividusToRecommend = set(indices.flat)
        
        setPostulantReel = set(df_utility.loc[df_utility['JOBOFFER_ID'] == offre]['INDIV_ID'])
        listesize.append(len(setIndividusToRecommend))
        if len(setIndividusToRecommend) != 0:
            listeResult.append(100*len(setPostulantReel.intersection(setIndividusToRecommend))/float(len(setIndividusToRecommend)))
        listeResult2.append(100*len(setPostulantReel.intersection(setIndividusToRecommend))/float(len(setPostulantReel)))
    print "Taille moyenne de la recommendation: %1.1f" % np.mean(listesize)
    print "Nombre d'offre test: %i" % len(listeOffre)
    print "Combien d'offres ont aboutis à une recommendation: %i" % len(listeResult)
    print "Précision de la recommendation: %1.2f" % np.mean(listeResult)
    print "Rappel de la recommendation: %1.2f" % np.mean(listeResult2)
    listPrecision.append(np.mean(listeResult))
    listRappel.append(np.mean(listeResult2))
    listTestSetSuccess.append(100*nbSuccessTestSet/float(len(listeOffre)))

print listPrecision
print listRappel
print listTestSetSuccess