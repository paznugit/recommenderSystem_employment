# -*- coding: utf-8 -*-
"""
Created on Tue Jan 05 13:40:03 2016

@author: IGPL3460
"""


import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds
from scipy.sparse.linalg import norm
from scipy import linalg as LA
import numpy as np
#from math import pow

# Parameter of this algorithm: The number of dimension used for SVD
k = 50
lambda_r = 0.0001
epsilon = 0.05
niter = 200
stoch = True

def loss_function(m, P, Q, lambda_r):
    """ fonction de cout"""
    (rows,cols) = m.nonzero()
    nb = len(rows)
    values = []
    for i in range(nb):
        values.append(P[rows[i],:].dot(Q[:,cols[i]]))

    return np.sum(np.power(m[rows,cols] - values,2)) + lambda_r*(np.sum(pow(P,2))+np.sum(pow(Q,2)))
    
def gr_loss_function(typeMatrice, id_val, m, P, Q, ndim, lambda_r):
    """ Gradient de la fonction de cout"""
    gr = 0
    if typeMatrice == "P":
        gr = np.zeros(ndim)
        for k in range(ndim):
            for o in m.getrow(id_val).indices:
                gr[k] += 2*(P[id_val,:].dot(Q[:,o])-m[id_val,o])*Q[k,o]
            gr[k] += 2*lambda_r*P[id_val,k]
    else:
        gr = np.zeros(ndim)
        for k in range(ndim):
            for i in m.getcol(id_val).indices:
                gr[k] += 2*(P[i,:].dot(Q[:,id_val])-m[i,id_val])*P[i,k]
            gr[k] += 2*lambda_r*Q[k,id_val]
    return gr
    
#==============================================================================
# Algorithme descente de gradient
#==============================================================================
def gradient(m, epsilon, lambda_r, niter, P_ini, Q_ini, ndim, lfun, gr_lfun, stoch=True):
    """ algorithme de descente du gradient:
        - m : matrice de donnée
        - epsilon : Facteur multiplicatif de descente
        - lambda_r : Facteur de regularisation
        - niter : nombre d'iterations
        - P_ini: P initial
        - Q_ini: Q initial
        - ndim: P ndim
        - lfun : fonction de cout
        - gr_lfun : gradient de la fonction de cout
        - stoch : True : gradient stochastique
        """
    #
    #w = np.zeros((niter, P_ini.size))
    (nx,ny) = m.shape
    #w[0] = w_ini
    loss = np.zeros(niter)
    loss[0] = lfun(m, P_ini, Q_ini,lambda_r)
        
    for i in range(1, niter):
        if stoch:
            idx = np.random.randint(nx)
            idy = np.random.randint(ny)
        else:
            # TODO
            idx = 0#$idx = np.arange(x.shape[0])
        P[idx,:] = P[idx,:] - epsilon*gr_lfun("P",idx,m,P,Q,ndim,lambda_r)
        Q[:,idy] = Q[:,idy] - epsilon*gr_lfun("Q",idy,m,P,Q,ndim,lambda_r)
        #w[i, :] = w[i - 1, :] - epsilon * gr_lfun(x[idx, :], y[idx], w[i - 1, :])
        loss[i] = lfun(m,P,Q,lambda_r)
    #return w, loss
    return P,Q,loss
    
# Extract the utility matrix (link between individual and job offer)
csv_input = '../input/dm_mec_21_ng.csv'
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
nbTrainSet = 1500
listCoordinateTestSet = []

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
        if i > nbTrainSet:
            break

nbTrainSet = len(listCoordinateTrainSet) 
print "Creation of train set OK"

shape = (nbIndiv, nbOffre)
'''shape = (6,5)
rows = [0,1,1,1,2,3,4,5,5]
cols = [4,0,1,4,2,0,0,1,3]
vals = [1,1,2,1,1,1,1,1,3]
m = coo_matrix((vals, (rows, cols)), shape=shape)
m = m.tocsr()
#print m.shape
P = np.array([[0,1],[0,1],[1,1],[1,2],[0,0],[0,2]])
Q = np.array([[1,1,0,1,1],[1,3,1,0,1]])
'''

m = coo_matrix((vals, (rows, cols)), shape=shape)
m = m.tocsr()

# SVD computation
print "Initialization of P and Q via SVD"
# Initialize the matrix using a singular value decomposition
u,s,vt = svds(m,k = k)
s = np.sqrt(s)
# We're now looking for P and Q such as R = P.Qt
P = (u.dot(np.diag(s)))
Q = (np.diag(s)).dot(vt)
print "Shape of P: %s" % str(P.shape)
print "Shape of Q: %s" % str(Q.shape)

P,Q,loss = gradient(m, epsilon, lambda_r, niter, P_ini = P, Q_ini = Q, ndim = k,
               lfun = loss_function, gr_lfun = gr_loss_function, stoch = stoch)

print np.sum(P)
print np.sum(Q)
print loss

'''nbSuccessTestSet = 0
nbSuccessTrainSet = 0


# Computation of prediction for train set
listTrainSetResult = []
print "Computation of success in train set"
nbTrainSet = nbmec - nbTestSet
print "nbTrainSet = %i" % nbTrainSet
for (indiv,offre) in listCoordinateTrainSet:
    prediction = P[indiv,:].dot(Q[:,offre])
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
    
# Computation of success in test set
print "Computation of success in test set"
print "nbTestSet = %i" % nbTestSet
for (indiv,offre) in listCoordinateTestSet:
    prediction = P[indiv,:].dot(Q[:,offre])
    if prediction > seuilSuccess:
        nbSuccessTestSet += 1
print "Computation of success in test set OK"
print "nbSuccessTestSet = %i" % nbSuccessTestSet
print "Taux de success Test Set: %1.1f" % (100*nbSuccessTestSet/float(nbTestSet))

listeProfile = []
# For each individual in test set, let's retrieve the list of job offer we would recommend
listNbRecommend = []
for (indiv,offre) in listCoordinateTestSet:
    if indiv in listeProfile:
        # Already done: We continue
        continue
    nbRecommend = 0
    for jobOfferId in range(nbOffre):
        # We don't want to look if the (indiv,offre) is in the train or test set
        if jobOfferId == offre:
            continue
        if (indiv,jobOfferId) in listCoordinateTrainSet:
            continue
        prediction = P[indiv,:].dot(Q[:,offre])
        if prediction > seuilSuccess:
            nbRecommend += 1  
    listNbRecommend.append(nbRecommend)
    listeProfile.append(indiv)
    
recomean = np.mean(listNbRecommend)
print "Nombre d'individus teste: %1.1f" % len(listNbRecommend)
print "Nombre de reco moyen par individu: %1.1f" % recomean
print "Taux de reco: %1.1f" % (100*recomean/float(nbOffre))'''


# Now let's look at a gradient descent to find P.Qt such as we always predict the
# appropriate value of the utility matrix when known


