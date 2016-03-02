# -*- coding: utf-8 -*-
"""
Created on Tue Jan 05 13:40:03 2016

@author: IGPL3460
"""


import pandas as pd
from scipy.sparse import coo_matrix
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.feature_extraction import DictVectorizer as DV
from scipy import spatial

ponderationMetier = True
number_neighbors = 20
seuil = 2
# List of norm: l1,l2,cosine
norm = 'cosine'

csv_input = '../input/dm_mec_21_ng_bo.csv'
csv_jobofferdict__input = '../input/joboffer_dict_21.csv'
csv_dmoff_input = '../input/dm_off_21_ng.csv'
csv_rome_affinity_input = '../input/affinity_Rome_matrix.csv'
    
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
    if i > 5:
            break
print "Creation of train set OK"

nbTrainSet = len(listCoordinateTrainSet) 

shape = (nbIndiv, nbOffre)
m = coo_matrix((vals, (rows, cols)), shape=shape)
m = m.tocsr()

# Let's retrieve the conversion dictionnary of job offer
df_convertJobOffer = pd.read_csv(csv_jobofferdict__input, names = ['KC_OFFRE_ID','JOBOFFER_ID'])
dictOffreConvert = dict(zip(list(df_convertJobOffer['JOBOFFER_ID']),list(df_convertJobOffer['KC_OFFRE_ID'])))

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

# Creation of a dictionnary Rome
setRome = set(pd.unique(df_offre['dc_rome_id']))
irome = 0
dictRome = {}
for rome in setRome:
    dictRome[rome] = irome
    irome += 1
    
# Extraction of the Rome affinity matrix
ROME1 = 'ROME1'
ROME2 = 'ROME2'
SCORE = 'SCORE'
df_rome_aff = pd.read_csv(csv_rome_affinity_input, names = [ROME1,ROME2,SCORE])
listeRome1 = list(df_rome_aff[ROME1]) 
listeRome2 = list(df_rome_aff[ROME2]) 
listeScore = list(df_rome_aff[SCORE]) 
dictRomeAffinity = {}
nbRomeNonPresent = 0
for i in range(len(listeRome1)):
    if listeRome1[i] not in dictRome:
        nbRomeNonPresent += 1
    elif listeRome2[i] not in dictRome:
        nbRomeNonPresent += 1
    else:
        rome1 = dictRome[listeRome1[i]]
        rome2 = dictRome[listeRome2[i]]
        dictRomeAffinity[(rome1,rome2)] = int(listeScore[i])
print "Rome non présent: %i" % nbRomeNonPresent

#==============================================================================
# computeDistanceBetweenJobOffer
# Function to compute the distance between 2 job offer
#==============================================================================
def computeDistanceBetweenJobOffer(jo1, jo2):
    rome1 = jo1[0]
    rome2 = jo2[0]
    if rome1 == rome2:
        return spatial.distance.cosine(jo1[1:], jo2[1:])
    if (rome2,rome1) in dictRomeAffinity:
        return (200-dictRomeAffinity[(rome2,rome1)])*0.01*(spatial.distance.cosine(jo1[1:], jo2[1:]))
    else:
        return 2

cols_to_retain = ['dc_naturecontrat_id', 'dc_typecontrat_id','dc_langue_1_id', 'dc_permis_1_id']
cols_to_retain = ['dn_frequencedeplacement']
cols_to_retain = ['dn_frequencedeplacement','dn_typedeplacement','dc_naturecontrat_id', 'dc_typecontrat_id','dc_langue_1_id', 'dc_permis_1_id','dc_qualification_id']

cat_df = df_offre[cols_to_retain].astype(str)
cat_dict = cat_df.T.to_dict().values()
vectorizer = DV( sparse = False )
x_offre = vectorizer.fit_transform(cat_dict)

listeSalaire = list(df_offre['dn_salaireannuelminimumeuros'])
scaler = StandardScaler(with_mean = False, with_std = True)
listeSalaire = scaler.fit_transform(listeSalaire)
#listeSalaire = preprocessing.minmax_scale(listeSalaire)
joboffer_content = np.hstack((x_offre,listeSalaire.reshape((len(listeSalaire), 1))))

#listRome = list(df_offre['dc_rome_id'])
if ponderationMetier:
    listRome = df_offre.apply(lambda x: dictRome[x['dc_rome_id']] , axis = 1)
    #listRome = np.array(list(df_offre['dc_rome_id']))
    joboffer_content = np.hstack((listRome.reshape((len(listRome), 1)),joboffer_content))

nbSuccessTestSet = 0
        
print "Iteration over each test set"
print "nbTestSet = %i" % nbTestSet

for number_neighbors in range(50,51):
    nbSuccessTestSet = 0
    listeResult = []
    listeResult2 = []
    listesize = []
    listeOffre = []
    for (indivId,joboffer_id) in listCoordinateTestSet:
         # We get the real offre_id
        kc_offre = dictOffreConvert[joboffer_id]
        offre = df_offre.loc[df_offre['kc_offre'] == kc_offre]
        indexoffre = offre.index[0]
        coderome = list(offre['dc_rome_id'])[0]
        codeapp = list(offre['dc_appelationrome_id'])[0]
        
        if ponderationMetier:
            job_offerToAnalyse = joboffer_content
            listeIndex = df_offre.index
        else: 
            df_offre_comparison = df_offre.loc[df_offre['dc_rome_id'] == coderome]
            #df_offre_comparison = df_offre.loc[df_offre['dc_appelationrome_id'] == codeapp]
            #listeIndex = np.delete(df_offre_comparison.index,indexoffre)
            listeIndex = list(df_offre_comparison.index)
            listeIndex = np.delete(listeIndex,listeIndex.index(indexoffre))
            job_offerToAnalyse = joboffer_content[listeIndex]
        #print job_offerToAnalyse
        if len(job_offerToAnalyse) < number_neighbors:
            k = len(job_offerToAnalyse)
        else:
            k = number_neighbors
        if len(job_offerToAnalyse) == 0:
            print "Find one empty: kc_offre = %s - Rome = %s" % (kc_offre,coderome)
            continue
        if ponderationMetier:
            nn = NearestNeighbors(n_neighbors=k, algorithm='brute', metric=lambda a,b: computeDistanceBetweenJobOffer(a,b)).fit(job_offerToAnalyse)
        else:
            nn = NearestNeighbors(n_neighbors=k, algorithm='brute', metric=norm).fit(job_offerToAnalyse)
        distances, indices = nn.kneighbors(joboffer_content[indexoffre])
         
        listeKcOffre = list(df_offre.iloc[listeIndex[indices[0]]]['kc_offre'])
        listeIdOffre = list(df_convertJobOffer.loc[df_convertJobOffer['KC_OFFRE_ID'].isin(listeKcOffre)]['JOBOFFER_ID'])
        df_result =  df_utility.loc[df_utility['JOBOFFER_ID'].isin(listeIdOffre)].groupby('INDIV_ID')['JOBOFFER_ID'].count().reset_index()
        setIndividusToRecommend = set(df_result.loc[df_result['JOBOFFER_ID'] >= seuil]['INDIV_ID'])
        if indivId in setIndividusToRecommend:
            nbSuccessTestSet += 1
            
        if joboffer_id in listeOffre:
            # Already done: We continue
            continue
        listeOffre.append(joboffer_id)
        setPostulantReel = set(df_utility.loc[df_utility['JOBOFFER_ID'] == joboffer_id]['INDIV_ID'])
        listesize.append(len(setIndividusToRecommend))
        if len(setIndividusToRecommend) != 0:
            listeResult.append(100*len(setPostulantReel.intersection(setIndividusToRecommend))/float(len(setIndividusToRecommend)))
        listeResult2.append(100*len(setPostulantReel.intersection(setIndividusToRecommend))/float(len(setPostulantReel)))
    
    print "k = %i" % number_neighbors
    print "nbSuccessTestSet = %i" % nbSuccessTestSet
    print "Taux de success Test Set: %1.1f" % (100*nbSuccessTestSet/float(nbTestSet))
    print "Taille moyenne de la recommendation: %1.1f" % np.mean(listesize)
    print "Nombre d'offre test: %i" % len(listeOffre)
    print "Combien d'offres ont aboutis à une recommendation: %i" % len(listeResult)
    print "Précision de la recommendation: %1.2f" % np.mean(listeResult)
    print "Rappel de la recommendation: %1.2f" % np.mean(listeResult2)