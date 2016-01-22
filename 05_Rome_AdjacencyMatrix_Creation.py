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
csv_input = '../input/dm_mec_52_ng_bo.csv'
csv_joboffer = '../input/dm_off_52_ng.csv'
csv_joboffer_dict = '../input/joboffer_dict_52.csv'
csv_cible = '../input/affinity_Rome_matrix_52.csv'

df_utility = pd.read_csv(csv_input)

# Extract the job offer
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

df_offre = pd.read_csv(csv_joboffer, names = columnNames)

listOffreId = list(df_offre['kc_offre'])
listRome = list(df_offre['dc_rome_id'])
nbOffre = len(listOffreId)

# Let's retrieve the conversion dictionnary of job offer
df_convertJobOffer = pd.read_csv(csv_joboffer_dict, names = ['KC_OFFRE_ID','JOBOFFER_ID'])
dictOffreConvert = dict(zip(list(df_convertJobOffer['KC_OFFRE_ID']),list(df_convertJobOffer['JOBOFFER_ID'])))

listRefRome = list(pd.unique(df_offre['dc_rome_id'].values))

dictRome = {}

# For each 'code Rome' we retrieve the list of users that that applied to
# a job offer link to this 'code Rome'
#listRefRome = ['J1505','J1506']
for rome in listRefRome:
    listJobOfferAssociated = []
    for i in range(nbOffre):
        if listRome[i] == rome:
            if listOffreId[i] in dictOffreConvert:
                listJobOfferAssociated.append(dictOffreConvert[listOffreId[i]])
            
    setUsers = set(list(df_utility.loc[df_utility['JOBOFFER_ID'].isin(listJobOfferAssociated)]['INDIV_ID']))
    dictRome[rome] = setUsers
    print "For Rome %s, number of application %i" % (rome,len(setUsers))

dict_r = {}
# Now let's create the adjacency matrix
for rome1 in listRefRome:
    for rome2 in listRefRome:
        if rome1 == rome2:
            # In the diagonal, we store the number of user that have applied to the job
            #dict_r[rome1,rome2] = len(set(dictRome[rome1]))
            rome1 = rome2
        else:
            setuser1 = dictRome[rome1]
            setuser2 = dictRome[rome2]
            nbuserIntersect = len(setuser1.intersection(setuser2))
            nbuser1 = len(setuser1)
            #nbuser2 = len(setuser2)
            #sumuser = nbuser1 + nbuser2
            if nbuser1 > 20:
                r = 100*nbuserIntersect/float(nbuser1)
                if r > 20:
                    dict_r[rome1,rome2] = r
        
with open(csv_cible, 'w') as outfile:
    for romes, r in dict_r.iteritems():
        outfile.write(romes[0])
        outfile.write(",")
        outfile.write(romes[1])
        outfile.write(",")
        outfile.write(("%1.0f"%r))
        outfile.write("\n")
    
# Algo:
# Pour chaque code ROME, générer une liste de user associé (via un dictionnaire)
# Pour la génération de l'adjacency matrix:
# Pour chaque cell code ROME 1 / code ROME 2 => Lancer une jaccard similarity, 
# ou quelque chose d'approchant. Et on a la valeur de la celle en divisant par le len
# de chaque liste associé à chaque code ROME
