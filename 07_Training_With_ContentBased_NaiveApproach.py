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


number_neighbors = 20
    
csv_input = '../input/dm_mec_21_ng_bo.csv'
csv_jobofferdict__input = '../input/joboffer_dict_21.csv'
csv_dmoff_input = '../input/dm_off_21_ng.csv'

#==============================================================================
# computeDistanceBetweenJobOffer
# Function to compute the distance between 2 job offer
#==============================================================================
def computeDistanceBetweenJobOffer(jo1, jo2):
    return 0
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
nbTestSet = 10
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

cols_to_retain = ['dc_naturecontrat_id', 'dc_typecontrat_id','dc_langue_1_id', 'dc_permis_1_id']      
cat_df = df_offre[cols_to_retain]
cat_dict = cat_df.T.to_dict().values()
vectorizer = DV( sparse = False )
x_offre = vectorizer.fit_transform(cat_dict)

listeSalaire = list(df_offre['dn_salaireannuelminimumeuros'])
scaler = StandardScaler(with_mean = False, with_std = True)
listeSalaire = scaler.fit_transform(listeSalaire)
#listeSalaire = preprocessing.minmax_scale(listeSalaire)
joboffer_content = np.hstack((x_offre,listeSalaire.reshape((len(listeSalaire), 1))))


nbSuccessTestSet = 0
nbSuccessTrainSet = 0

# Iterate over each job offer to test
print "Iteration over each train set"
print "nbTrainSet = %i" % nbTrainSet
for (indivId,joboffer_id) in listCoordinateTrainSet:
     # We get the real offre_id
    kc_offre = dictOffreConvert[joboffer_id]
    offre = df_offre.loc[df_offre['kc_offre'] == kc_offre]
    indexoffre = offre.index[0]
    coderome = list(offre['dc_rome_id'])[0]
    codeapp = list(offre['dc_appelationrome_id'])[0]
    #print "#######################################"
    #print "We deal wih job offer %s: %s %s" % (kc_offre,coderome,str(codeapp))
    
    df_offre_comparison = df_offre.loc[df_offre['dc_rome_id'] == coderome]
    #df_offre_comparison = df_offre.loc[df_offre['dc_appelationrome_id'] == codeapp]
    listeIndex = df_offre_comparison.index
    
    job_offerToAnalyse = joboffer_content[listeIndex]

    if len(job_offerToAnalyse) < 20:
        number_neighbors = len(job_offerToAnalyse)
    nn = NearestNeighbors(n_neighbors=number_neighbors, algorithm='brute', metric='cosine').fit(job_offerToAnalyse)
    distances, indices = nn.kneighbors(joboffer_content[indexoffre])
     
    listeKcOffre = list(df_offre.iloc[listeIndex[indices[0]]]['kc_offre'])
    listeIdOffre = list(df_convertJobOffer.loc[df_convertJobOffer['KC_OFFRE_ID'].isin(listeKcOffre)]['JOBOFFER_ID'])
    df_result =  df_utility.loc[df_utility['JOBOFFER_ID'].isin(listeIdOffre)].groupby('INDIV_ID')['JOBOFFER_ID'].count().reset_index()
    setIndividusToRecommend = set(df_result.loc[df_result['JOBOFFER_ID'] >= 2]['INDIV_ID'])
    if indivId in setIndividusToRecommend:
        nbSuccessTrainSet += 1
        
print "nbSuccessTrainSet = %i" % nbSuccessTrainSet
print "Taux de success Train Set: %1.1f" % (100*nbSuccessTrainSet/float(nbTrainSet))
        
print "Iteration over each test set"
print "nbTestSet = %i" % nbTestSet
listeResult = []
listesize = []
for (indivId,joboffer_id) in listCoordinateTestSet:
     # We get the real offre_id
    kc_offre = dictOffreConvert[joboffer_id]
    offre = df_offre.loc[df_offre['kc_offre'] == kc_offre]
    indexoffre = offre.index[0]
    coderome = list(offre['dc_rome_id'])[0]
    codeapp = list(offre['dc_appelationrome_id'])[0]
    '''print "#######################################"
    print "We deal wih job offer %s: %s %s" % (kc_offre,coderome,str(codeapp))'''
    
    df_offre_comparison = df_offre.loc[df_offre['dc_rome_id'] == coderome]
    #df_offre_comparison = df_offre.loc[df_offre['dc_appelationrome_id'] == codeapp]
    listeIndex = df_offre_comparison.index
    
    job_offerToAnalyse = joboffer_content[listeIndex]

    if len(job_offerToAnalyse) < 20:
        number_neighbors = len(job_offerToAnalyse)
    nn = NearestNeighbors(n_neighbors=number_neighbors, algorithm='brute', metric='cosine').fit(job_offerToAnalyse)
    distances, indices = nn.kneighbors(joboffer_content[indexoffre])
     
    listeKcOffre = list(df_offre.iloc[listeIndex[indices[0]]]['kc_offre'])
    listeIdOffre = list(df_convertJobOffer.loc[df_convertJobOffer['KC_OFFRE_ID'].isin(listeKcOffre)]['JOBOFFER_ID'])
    df_result =  df_utility.loc[df_utility['JOBOFFER_ID'].isin(listeIdOffre)].groupby('INDIV_ID')['JOBOFFER_ID'].count().reset_index()
    setIndividusToRecommend = set(df_result.loc[df_result['JOBOFFER_ID'] >= 2]['INDIV_ID'])
    if indivId in setIndividusToRecommend:
        nbSuccessTestSet += 1
        
    setPostulantReel = set(df_utility.loc[df_utility['JOBOFFER_ID'] == joboffer_id]['INDIV_ID'])
    listesize.append(len(setIndividusToRecommend))
    if len(setIndividusToRecommend) != 0:
        print setIndividusToRecommend
        print setPostulantReel
        listeResult.append(len(setPostulantReel.intersection(setIndividusToRecommend))/float(len(setIndividusToRecommend)))
    
print "nbSuccessTestSet = %i" % nbSuccessTestSet
print "Taux de success Test Set: %1.1f" % (100*nbSuccessTestSet/float(nbTestSet))
print "Taille moyenne de la recommendation: %1.1f" % np.mean(listesize)
print "Combien d'offres ont aboutis à une recommendation: %i" % len(listeResult)
print "Adéquation moyen avec la réalité: %1.2f" % np.mean(listeResult)

'''listeResult = []
listesize = []
for joboffer_id in listJobOfferToTest:
    # We get the real offre_id
    kc_offre = dictOffreConvert[joboffer_id]
    offre = df_offre.loc[df_offre['kc_offre'] == kc_offre]
    indexoffre = offre.index[0]
    coderome = list(offre['dc_rome_id'])[0]
    codeapp = list(offre['dc_appelationrome_id'])[0]
    print "#######################################"
    print "We deal wih job offer %s: %s %s" % (kc_offre,coderome,str(codeapp))
    
    df_offre_comparison = df_offre.loc[df_offre['dc_rome_id'] == coderome]
    #df_offre_comparison = df_offre.loc[df_offre['dc_appelationrome_id'] == codeapp]
    listeIndex = df_offre_comparison.index
    
    job_offerToAnalyse = joboffer_content[listeIndex]

    #print job_offerToAnalyse
    if len(job_offerToAnalyse) < 20:
        number_neighbors = len(job_offerToAnalyse)
    nn = NearestNeighbors(n_neighbors=number_neighbors, algorithm='brute', metric='cosine').fit(job_offerToAnalyse)
    distances, indices = nn.kneighbors(joboffer_content[indexoffre])
     
    listeKcOffre = list(df_offre.iloc[listeIndex[indices[0]]]['kc_offre'])
    listeIdOffre = list(df_convertJobOffer.loc[df_convertJobOffer['KC_OFFRE_ID'].isin(listeKcOffre)]['JOBOFFER_ID'])
    df_result =  df_utility.loc[df_utility['JOBOFFER_ID'].isin(listeIdOffre)].groupby('INDIV_ID')['JOBOFFER_ID'].count().reset_index()
    setIndividusToRecommend = set(df_result.loc[df_result['JOBOFFER_ID'] >= 2]['INDIV_ID'])
    #print "Liste des candidats auxquels on veut recommender l'offre:"
    #print setIndividusToRecommend
    #print df_utility.loc[df_utility['INDIV_ID'].isin(listeIndividusToRecommend)]
    setPostulantReel = set(df_utility.loc[df_utility['JOBOFFER_ID'] == joboffer_id]['INDIV_ID'])
    #print "Liste des candidats ayant réellement postulé:"
    #print setPostulantReel
    listesize.append(len(setIndividusToRecommend))
    if len(setIndividusToRecommend) != 0:
        listeResult.append(len(setPostulantReel.intersection(setIndividusToRecommend))/float(len(setIndividusToRecommend)))'''

