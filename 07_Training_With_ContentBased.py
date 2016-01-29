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
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.feature_extraction import DictVectorizer as DV
from scipy import spatial

csv_input = '../input/dm_mec_21_ng_bo.csv'
csv_jobofferdict__input = '../input/joboffer_dict_21.csv'
csv_dmoff_input = '../input/dm_off_21_ng.csv'
csv_rome_affinity_input = '../input/affinity_Rome_matrix.csv'

# Extraction of the Rome affinity matrix
ROME1 = 'ROME1'
ROME2 = 'ROME2'
SCORE = 'SCORE'
df_rome_aff = pd.read_csv(csv_rome_affinity_input, names = [ROME1,ROME2,SCORE])
listeRome1 = list(df_rome_aff[ROME1]) 
listeRome2 = list(df_rome_aff[ROME2]) 
listeScore = list(df_rome_aff[SCORE]) 
dictRomeAffinity = {}
for i in range(len(listeRome1)):
    rome1= listeRome1[i]
    rome2 = listeRome2[i]
    if rome2 in dictRomeAffinity:
        dictRomeAffinity[rome2].append((rome1,listeScore[i]))
    else:
        dictRomeAffinity[rome2] = [(rome1,listeScore[i])]
        
#==============================================================================
# computeDistanceBetweenJobOffer
# Function to compute the distance between profile:
# - userProfile
# - jobOfferProfile
# - List of codeRome of the userProfile
# - codeRome of the jobOfferProfile
#==============================================================================
def computeSimilarityBetweenJobOffer(userProfile,jobOfferProfile,listeCodeRome,codeRome, listGeo, geo):
    similarity = 0
    if codeRome in listeCodeRome:
        if geo in listGeo:
            similarity = spatial.distance.cosine(userProfile, jobOfferProfile)
    return similarity
      

    
# Extract the job offer
print "Extract the file of job offer and vectorize it"
columnNames = ['kc_offre','dn_frequencedeplacement','dn_typedeplacement',
               'dc_typexperienceprof_id','experienceMois','dc_rome_id',
               'dc_appelationrome_id','dc_naturecontrat_id',
               'dc_typecontrat_id','dureeContratJour',
               'dn_salaireannuelminimumeuros','dc_naf2','dc_qualification_id',
               'dc_modepresentation_emp_id','dc_langue_1_id',
               'dc_niveaulangue_1_id','dc_exigibilitelangue_1_id',
               'dc_permis_1_id','dc_exigibilitepermis_1_id',
               'dc_communelieutravail','dc_departementlieutravail',
               'dc_typelieutravail','dc_lbllieutravail']
df_offre = pd.read_csv(csv_dmoff_input, names = columnNames)

dictKcOffre_IndexOffre = dict(zip(list(df_offre['kc_offre']),df_offre.index))

cols_to_retain = ['dc_typexperienceprof_id', 'dc_naturecontrat_id', 'dc_typecontrat_id',
                  'dc_langue_1_id', 'dc_permis_1_id']      
cat_df = df_offre[cols_to_retain]
cat_dict = cat_df.T.to_dict().values()
vectorizer = DV( sparse = False )
x_offre = vectorizer.fit_transform(cat_dict)

# TODO: experienceMois
# dureeContratJour

listeSalaire = list(df_offre['dn_salaireannuelminimumeuros'])
scaler = StandardScaler(with_mean = False, with_std = True)
listeSalaire = scaler.fit_transform(listeSalaire)
#listeSalaire = preprocessing.minmax_scale(listeSalaire)
joboffer_content = np.hstack((x_offre,listeSalaire.reshape((len(listeSalaire), 1)))) 
    
listeCodeRome = np.array(df_offre['dc_rome_id'])
joboffer_content = np.hstack((joboffer_content,listeCodeRome.reshape((len(listeCodeRome), 1)))) 

listeGeo = np.array(df_offre['dc_departementlieutravail'])
joboffer_content = np.hstack((joboffer_content,listeGeo.reshape((len(listeGeo), 1)))) 

# Extract the utility matrix (link between individual and job offer)
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

# Let's retrieve the conversion dictionnary of job offer
df_convertJobOffer = pd.read_csv(csv_jobofferdict__input, names = ['KC_OFFRE_ID','JOBOFFER_ID'])
dictOffreConvert = dict(zip(list(df_convertJobOffer['JOBOFFER_ID']),list(df_convertJobOffer['KC_OFFRE_ID'])))

nbSuccessTestSet = 0
nbSuccessTrainSet = 0

# Computation of prediction for train set
listTrainSetResult = []
nbTrainSet = len(listCoordinateTrainSet)
print "nbTrainSet = %i" % nbTrainSet
for (indivId,jobOfferId) in listCoordinateTrainSet:
    # Retrieval of the list of job offers and computation of the user profile
    listJobOfferId = df_utility.loc[df_utility['INDIV_ID'] == indivId]['JOBOFFER_ID']
    nboffre = 0
    first = True
    listeCodeRome = []
    listeGeo = []
    '''print "######################"
    print "IndivId: %i" % indivId
    print "A postulé à:"'''
    for jobOfferId2 in listJobOfferId:
        if jobOfferId == jobOfferId2:
            continue
        nboffre += 1
        kc_offre_id = dictOffreConvert[jobOfferId2]
        indexOffre = dictKcOffre_IndexOffre[kc_offre_id]
        '''print (df_offre.loc[df_offre['kc_offre'] == kc_offre_id]).as_matrix()'''
        # We retrieve the job profile, everything but the last columns (code Rome)
        if first:
            profile = joboffer_content[indexOffre][:-2]
            first = False
        else:
            profile += joboffer_content[indexOffre][:-2]
        listeCodeRome.append(joboffer_content[indexOffre][-2])
        listeGeo.append(joboffer_content[indexOffre][-1])
    if nboffre == 0:
        # Shouldn't happen... but just in case
        print "Division by zero!"
        break
    userProfile = profile/float(nboffre)
    '''print "-----------------"
    print "UserProfile:"
    print userProfile'''
    # Let's see if we would recommend this job offer:
    kc_offre_id = dictOffreConvert[jobOfferId]
    '''print "Calcul pour cette offre:"
    print (df_offre.loc[df_offre['kc_offre'] == kc_offre_id]).as_matrix()'''
    indexOffre = dictKcOffre_IndexOffre[kc_offre_id]
    jobOfferProfile = joboffer_content[indexOffre][:-2]
    codeRome = joboffer_content[indexOffre][-2]
    geo = joboffer_content[indexOffre][-1]
    sim = computeSimilarityBetweenJobOffer(userProfile,jobOfferProfile,listeCodeRome,codeRome,listeGeo,geo)
    '''print "Similarity avec userProfile: %1.3f" % sim'''
    listTrainSetResult.append(sim)

# Computation of seuilSuccess to have about 90% of success in train set
#print listTrainSetResult
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
for (indivId,jobOfferId) in listCoordinateTestSet:
    # Retrieval of the list of job offers and computation of the user profile
    listJobOfferId = df_utility.loc[df_utility['INDIV_ID'] == indivId]['JOBOFFER_ID']
    nboffre = 0
    first = True
    listeCodeRome = []
    listeGeo = []
    for jobOfferId2 in listJobOfferId:
        if jobOfferId == jobOfferId2:
            continue
        nboffre += 1
        kc_offre_id = dictOffreConvert[jobOfferId2]
        indexOffre = dictKcOffre_IndexOffre[kc_offre_id]
        # We retrieve the job profile, everything but the last columns (code Rome)
        if first:
            profile = joboffer_content[indexOffre][:-2]
            first = False
        else:
            profile += joboffer_content[indexOffre][:-2]
        listeCodeRome.append(joboffer_content[indexOffre][-2])
        listeGeo.append(joboffer_content[indexOffre][-1])
    if nboffre == 0:
        # Shouldn't happen... but just in case
        print "Division by zero!"
        break
    userProfile = profile/float(nboffre)
    #dictIndivProfile[indivId] = profile/float(nboffre)
    # Let's see if we would recommend this job offer:
    kc_offre_id = dictOffreConvert[jobOfferId]
    indexOffre = dictKcOffre_IndexOffre[kc_offre_id]
    jobOfferProfile = joboffer_content[indexOffre][:-2]
    codeRome = joboffer_content[indexOffre][-2]
    geo = joboffer_content[indexOffre][-1]
    similarity = computeSimilarityBetweenJobOffer(userProfile,jobOfferProfile,listeCodeRome,codeRome,listeGeo,geo)

    if similarity > seuilSuccess:
        nbSuccessTestSet += 1
print "Computation of success in test set OK"
print "nbSuccessTestSet = %i" % nbSuccessTestSet
print "Taux de success Test Set: %1.1f" % (100*nbSuccessTestSet/float(nbTestSet))

listeProfile = []
# For each individual in test set, let's retrieve the list of job offer we would recommend
listNbRecommend = []
for (indivId,jobOfferId) in listCoordinateTestSet:
    if indivId in listeProfile:
        # Already done: We continue
        continue
    
    # Retrieval of the list of job offers and computation of the user profile
    listJobOfferId = df_utility.loc[df_utility['INDIV_ID'] == indivId]['JOBOFFER_ID']
    nboffre = 0
    first = True
    listeCodeRome = []
    listeGeo = []
    for jobOfferId2 in listJobOfferId:
        nboffre += 1
        kc_offre_id = dictOffreConvert[jobOfferId2]
        indexOffre = dictKcOffre_IndexOffre[kc_offre_id]
        # We retrieve the job profile, everything but the last columns (code Rome)
        if first:
            profile = joboffer_content[indexOffre][:-2]
            first = False
        else:
            profile += joboffer_content[indexOffre][:-2]
        listeCodeRome.append(joboffer_content[indexOffre][-2])
        listeGeo.append(joboffer_content[indexOffre][-1])
    # We then have the profile!!
    userProfile = profile/float(nboffre)   
    listeProfile.append(indivId)
    
    # Then we look for recommendation!
    nbRecommend = 0
    for jobOfferId2 in range(nbOffre):
        # We don't want to look if the (indiv,offre) is in the train or test set
        if jobOfferId2 in listJobOfferId:
            continue
        
        kc_offre_id = dictOffreConvert[jobOfferId2]
        indexOffre = dictKcOffre_IndexOffre[kc_offre_id]   
        
        jobOfferProfile = joboffer_content[indexOffre]
        codeRome = jobOfferProfile[-2]
        geo = jobOfferProfile[-1]
        similarity = computeSimilarityBetweenJobOffer(userProfile, jobOfferProfile[:-2],listeCodeRome,codeRome,listeGeo,geo)
        if similarity > seuilSuccess:
            nbRecommend += 1
 
    listNbRecommend.append(nbRecommend)
   
recomean = np.mean(listNbRecommend)
print "Nombre d'individus teste: %1.1f" % len(listNbRecommend)
print "Nombre de reco moyen par individu: %1.1f" % recomean
print "Taux de reco: %1.1f" % (100*recomean/float(nbOffre))


















'''for (indivId,jobOfferId) in listCoordinateTestSet:
    # Retrieval of the list of job offers and computation of the user profile
    listJobOfferId = df_utility.loc[df_utility['INDIV_ID'] == indivId]['JOBOFFER_ID']
    nboffre = 0
    first = True
    listeCodeRome = []
    listeGeo = []
    for jobOfferId2 in listJobOfferId:
        if jobOfferId == jobOfferId2:
            continue
        nboffre += 1
        kc_offre_id = dictOffreConvert[jobOfferId2]
        indexOffre = dictKcOffre_IndexOffre[kc_offre_id]
        # We retrieve the job profile, everything but the last columns (code Rome)
        if first:
            profile = joboffer_content[indexOffre][:-2]
            first = False
        else:
            profile += joboffer_content[indexOffre][:-2]
        listeCodeRome.append(joboffer_content[indexOffre][-2])
        listeGeo.append(joboffer_content[indexOffre][-1])
    userProfile = profile/float(nboffre)
    #dictIndivProfile[indivId] = profile/float(nboffre)
    # Let's see if we would recommend this job offer:
    kc_offre_id = dictOffreConvert[jobOfferId]
    indexOffre = dictKcOffre_IndexOffre[kc_offre_id]
    jobOfferProfile = joboffer_content[indexOffre][:-2]
    codeRome = joboffer_content[indexOffre][-2]
    geo = joboffer_content[indexOffre][-1]
    similarity = computeSimilarityBetweenJobOffer(userProfile,jobOfferProfile,listeCodeRome,codeRome,listeGeo,geo)

    if similarity > seuil:
        nbsuccess += 1
        
    index = 0
    for jobOfferProfile in joboffer_content:
        codeRome = jobOfferProfile[-2]
        geo = jobOfferProfile[-1]
        if index == indexOffre:
            continue
        nbOtherTested += 1
        similarity = computeSimilarityBetweenJobOffer(userProfile, jobOfferProfile[:-2],listeCodeRome,codeRome,listeGeo,geo)
        if similarity > seuil:
            nbOtherSuccess += 1
        index += 1
            
print "Nombre de test data correctement recommendé: %i" % nbsuccess
print "Nombre de résultats positif a cote: %i" % nbOtherSuccess
print "Sur combien: %i" % nbOtherTested'''