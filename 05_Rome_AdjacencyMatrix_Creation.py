# -*- coding: utf-8 -*-
"""
Created on Tue Jan 05 13:40:03 2016

@author: IGPL3460
"""

import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds
#from scipy import linalg as LA
import numpy as np



# Extract the utility matrix (link between individual and job offer)
csv_input = '../input/dm_mec_ng_bo.csv'
df_utility = pd.read_csv(csv_input)

# Extract the utility matrix (link between individual and job offer)
csv_input = '../input/dm_off_ng.csv'
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

df_offre = pd.read_csv(csv_input, names = columnNames)

listOffreId = list(df_offre['kc_offre'])
listRome = list(df_offre['dc_rome_id'])
nbOffre = len(listOffreId)

#dictOffre = dict(zip(listOffreId,listRome))

listRefRome = list(pd.unique(df_offre['dc_rome_id'].values))

dictRome = {}

# For each 'code Rome' we retrieve the list of users that that applied to
# a job offer link to this 'code Rome'
for rome in listRefRome:
    listJobOfferAssociated = []
    for i in range(nbOffre):
        if listRome[i] == rome:
            listJobOfferAssociated.append(listOffreId[i])
            
    listUsers = list(df_utility.loc[df_utility['JOBOFFER_ID'].isin(listJobOfferAssociated)]['INDIV_ID'])
    dictRome[rome] = listUsers
   
# Now let's create the adjacency matrix
for rome1 in listRefRome:
    for rome2 in listRefRome:
        if rome1 == rome2:
            
    
# Algo:
# Pour chaque code ROME, générer une liste de user associé (via un dictionnaire)
# Pour la génération de l'adjacency matrix:
# Pour chaque cell code ROME 1 / code ROME 2 => Lancer une jaccard similarity, 
# ou quelque chose d'approchant. Et on a la valeur de la celle en divisant par le len
# de chaque liste associé à chaque code ROME
