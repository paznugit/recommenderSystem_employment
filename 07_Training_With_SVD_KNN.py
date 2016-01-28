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
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds
from scipy import linalg as LA
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.feature_extraction import DictVectorizer as DV

# Parameter of this algorithm: The number of dimension used for SVD
k = 800
number_neighbors = 20

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

nbTrainSet = len(listCoordinateTrainSet)  

print "Computation of SVD"
# Initialize the matrix using a singular value decomposition
u,s,vt = svds(m,k = k)
print "Computation of SVD OK"
s = np.sqrt(s)
# We're now looking for P and Q such as R = P.Qt
P = (u.dot(np.diag(s)))
Q = (np.diag(s)).dot(vt)
print "Shape of P: %s" % str(P.shape)
print "Shape of Q: %s" % str(Q.shape)

# creation of the neghbour space
nbSuccessTestSet = 0
nbSuccessTrainSet = 0

nn = NearestNeighbors(n_neighbors=number_neighbors, algorithm='brute', metric='cosine').fit(P)

# Computation in train set
for (indiv,offre) in listCoordinateTrainSet:
    # Let's retrieve the 20 more similar userProfile based on the jobofferProfile
    distances, indices = nn.kneighbors(Q[offre])
    
    if indiv in indices:
        nbSuccessTrainSet += 1
              
print "nbTrainSet = %i" % nbTrainSet
print "nbSuccessTrainSet = %i" % nbSuccessTrainSet
print "Taux de success Train Set: %1.1f" % (100*nbSuccessTrainSet/float(nbTrainSet))

# Computation in test set
for (indiv,offre) in listCoordinateTestSet:
    #prediction = P[indiv,:].dot(Q[:,offre])
    # Let's retrieve the 20 more similar userProfile based on the jobofferProfile
    distances, indices = nn.kneighbors(Q[offre])
    
    if indiv in indices:
        nbSuccessTestSet += 1
        
print "nbTestSet = %i" % nbTestSet
print "nbSuccessTestSet = %i" % nbSuccessTestSet
print "Taux de success Test Set: %1.1f" % (100*nbSuccessTestSet/float(nbTestSet))



'''
    df_result = df_utility.iloc[indices[0]].groupby('JOBOFFER_ID')['INDIV_ID'].count().reset_index()
    setJobOfferToRecommend = set(df_result.loc[df_result['INDIV_ID'] >= 2]['JOBOFFER_ID'])  
    setJobOfferReel = set(df_utility.loc[df_utility['INDIV_ID'] == indiv]['JOBOFFER_ID'])
    
    listesize.append(len(setJobOfferToRecommend))
    if len(setJobOfferToRecommend) != 0:
        listeResult.append(len(setJobOfferReel.intersection(setJobOfferToRecommend))/float(len(setJobOfferToRecommend)))
        
    if offre in setJobOfferToRecommend:
        nbsuccess += 1


print "Nombre de succès: %i" % nbsuccess
print "Taille moyenne de la recommendation: %1.1f" % np.mean(listesize)
print "Combien d'individus ont aboutis à une recommendation: %i" % len(listeResult)
print "Adéquation moyen avec la réalité: %1.2f" % np.mean(listeResult)'''


