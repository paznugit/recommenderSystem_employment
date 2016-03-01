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

ponderationMetier = False

csv_input = '../input/dm_mec_21_ng_bo.csv'
csv_jobofferdict__input = '../input/joboffer_dict_21.csv'
csv_dmoff_input = '../input/dm_off_21_ng.csv'
csv_rome_affinity_input = '../input/affinity_Rome_matrix.csv'
        
#==============================================================================
# computeSimilarityBetweenJobOffer
# Function to compute the similarity between profile:
# - userProfile
# - jobOfferProfile
# - List of codeRome of the userProfile
# - codeRome of the jobOfferProfile
#==============================================================================
def computeSimilarityBetweenJobOffer(userProfile_content,userProfile_rome,userProfile_geo,
                                     joboffer_content,joboffer_rome,joboffer_geo):
    similarity = 1-spatial.distance.cosine(userProfile_content, joboffer_content)
    similarity *= (1-spatial.distance.cosine(userProfile_rome, joboffer_rome))
    similarity *= (1-spatial.distance.cosine(userProfile_geo, joboffer_geo))
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
  
cols_to_retain = ['dn_frequencedeplacement','dn_typedeplacement','dc_naturecontrat_id',
                  'dc_typecontrat_id','dc_langue_1_id', 'dc_permis_1_id','dc_qualification_id']

cat_df = df_offre[cols_to_retain].astype(str)
cat_dict = cat_df.T.to_dict().values()
vectorizer = DV( sparse = False )
x_offre = vectorizer.fit_transform(cat_dict)

listeSalaire = list(df_offre['dn_salaireannuelminimumeuros'])
scaler = StandardScaler(with_mean = False, with_std = True)
listeSalaire = scaler.fit_transform(listeSalaire)
joboffer_content = np.hstack((x_offre,listeSalaire.reshape((len(listeSalaire), 1)))) 

df_rome = pd.get_dummies(df_offre['dc_rome_id'])
listeLibRome = df_rome.columns.values
joboffer_rome = df_rome.as_matrix()

if ponderationMetier:
    # Extraction of the Rome affinity matrix
    nbRome = len(listeLibRome)
    listeIndexRome = range(nbRome)
    dictIndexRome = dict(zip(listeLibRome,listeIndexRome)) 
    
    romeAffMatrix = np.zeros((nbRome,nbRome))
    ROME1 = 'ROME1'
    ROME2 = 'ROME2'
    SCORE = 'SCORE'
    df_rome_aff = pd.read_csv(csv_rome_affinity_input, names = [ROME1,ROME2,SCORE])
    listeRome1 = list(df_rome_aff[ROME1]) 
    listeRome2 = list(df_rome_aff[ROME2]) 
    listeScore = list(df_rome_aff[SCORE]) 
    for i in range(len(listeRome1)):
        rome1 = listeRome1[i]
        rome2 = listeRome2[i]
        if rome1 in dictIndexRome:
            if rome2 in dictIndexRome:
               romeAffMatrix[dictIndexRome[rome2],dictIndexRome[rome1]] = listeScore[i]
    # Creation of the new Rome Matrix
    joboffer_rome = joboffer_rome.dot(romeAffMatrix)

df_geo = pd.get_dummies(df_offre['dc_departementlieutravail'])
joboffer_geo = df_geo.as_matrix()

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

# Computation of prediction for train set
listTrainSetResult = []
nbTrainSet = len(listCoordinateTrainSet)
print "nbTrainSet = %i" % nbTrainSet
for (indivId,jobOfferId) in listCoordinateTrainSet:
    # Retrieval of the list of job offers and computation of the user profile
    listJobOfferId = df_utility.loc[df_utility['INDIV_ID'] == indivId]['JOBOFFER_ID']
    nboffre = 0
    first = True

    for jobOfferId2 in listJobOfferId:
        if jobOfferId == jobOfferId2:
            continue
        nboffre += 1
        kc_offre_id = dictOffreConvert[jobOfferId2]
        indexOffre = dictKcOffre_IndexOffre[kc_offre_id]
        # We retrieve the job profile, everything but the last columns (code Rome)
        if first:
            profile_content = joboffer_content[indexOffre]
            profile_rome = joboffer_rome[indexOffre]
            profile_geo = joboffer_geo[indexOffre]
            first = False
        else:
            profile_content += joboffer_content[indexOffre]
            profile_rome += joboffer_rome[indexOffre]
            profile_geo += joboffer_geo[indexOffre]
    if nboffre == 0:
        # Shouldn't happen... but just in case
        print "Division by zero!"
        break
    #print profile
    userProfile_content = profile_content/float(nboffre)
    userProfile_rome = profile_rome/float(nboffre)
    userProfile_geo = profile_geo/float(nboffre)
    # Let's see if we would recommend this job offer:
    kc_offre_id = dictOffreConvert[jobOfferId]
    indexOffre = dictKcOffre_IndexOffre[kc_offre_id]
    sim = computeSimilarityBetweenJobOffer(userProfile_content,userProfile_rome,userProfile_geo,
                                            joboffer_content[indexOffre],joboffer_rome[indexOffre],joboffer_geo[indexOffre])
    listTrainSetResult.append(sim)

# Computation of seuilSuccess to have about 90% of success in train set
q = 10
seuilSuccess = float(np.percentile(listTrainSetResult,q))
print "Seuil de succes positionne a: %1.5f" % seuilSuccess

nbSuccessTrainSet = 0
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

nbSuccessTestSet = 0
listeResult = []
listeResult2 = []
listesize = []
listeOffre = []
for (indivId,jobOfferId) in listCoordinateTestSet:
    
    if jobOfferId in listeOffre:
        # Already done: We continue
        continue
    listeOffre.append(jobOfferId)
    
    setPostulantReel = set(df_utility.loc[df_utility['JOBOFFER_ID'] == jobOfferId]['INDIV_ID'])
    setIndividusToRecommend = set()
    
    # Creation of profile for each individual
    for indivId2 in range(nbIndiv):
        # Retrieval of the list of job offers and computation of the user profile
        listJobOfferId = df_utility.loc[df_utility['INDIV_ID'] == indivId]['JOBOFFER_ID']
        
        nboffre = 0
        first = True
    
        for jobOfferId2 in listJobOfferId:
            if jobOfferId == jobOfferId2:
                continue
            nboffre += 1
            kc_offre_id = dictOffreConvert[jobOfferId2]
            indexOffre = dictKcOffre_IndexOffre[kc_offre_id]
            # We retrieve the job profile, everything but the last columns (code Rome)
            if first:
                profile_content = joboffer_content[indexOffre]
                profile_rome = joboffer_rome[indexOffre]
                profile_geo = joboffer_geo[indexOffre]
                first = False
            else:
                profile_content += joboffer_content[indexOffre]
                profile_rome += joboffer_rome[indexOffre]
                profile_geo += joboffer_geo[indexOffre]
        if nboffre == 0:
            # Shouldn't happen... but just in case
            print "Division by zero!"
            break
        userProfile_content = profile_content/float(nboffre)
        userProfile_rome = profile_rome/float(nboffre)
        userProfile_geo = profile_geo/float(nboffre)
        # Let's see if we would recommend this job offer:
        kc_offre_id = dictOffreConvert[jobOfferId]
        indexOffre = dictKcOffre_IndexOffre[kc_offre_id]
        sim = computeSimilarityBetweenJobOffer(userProfile_content,userProfile_rome,userProfile_geo,
                                                joboffer_content[indexOffre],joboffer_rome[indexOffre],joboffer_geo[indexOffre])
        
        if sim > seuilSuccess:
            setIndividusToRecommend.add(indivId2)
        
    listesize.append(len(setIndividusToRecommend))
    if len(setIndividusToRecommend) != 0:
        listeResult.append(100*len(setPostulantReel.intersection(setIndividusToRecommend))/float(len(setIndividusToRecommend)))
    listeResult2.append(100*len(setPostulantReel.intersection(setIndividusToRecommend))/float(len(setPostulantReel)))

print "nbSuccessTestSet = %i" % nbSuccessTestSet
print "Taux de success Test Set: %1.1f" % (100*nbSuccessTestSet/float(len(listeOffre)))
print "Taille moyenne de la recommendation: %1.1f" % np.mean(listesize)
print "Nombre d'offre test: %i" % len(listeOffre)
print "Combien d'offres ont aboutis à une recommendation: %i" % len(listeResult)
print "Précision de la recommendation: %1.2f" % np.mean(listeResult)
print "Rappel de la recommendation: %1.2f" % np.mean(listeResult2)