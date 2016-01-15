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
csv_input = '../input/dm_mec_21_ng_bo.csv'
df_utility = pd.read_csv(csv_input)

# Extract the job offer
csv_input = '../input/dm_off_21_ng.csv'
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
listApp = list(df_offre['dc_appelationrome_id'])
nbOffre = len(listOffreId)

# Let's retrieve the conversion dictionnary of job offer
csv_input = '../input/joboffer_dict_21.csv'
df_convertJobOffer = pd.read_csv(csv_input, names = ['KC_OFFRE_ID','JOBOFFER_ID'])
dictOffreConvert = dict(zip(list(df_convertJobOffer['KC_OFFRE_ID']),list(df_convertJobOffer['JOBOFFER_ID'])))

listRefApp= list(pd.unique(df_offre['dc_appelationrome_id'].values))

dictRome = {}

# For each 'code Rome' we retrieve the list of users that that applied to
# a job offer link to this 'code Rome'
#listRefRome = ['J1505','J1506']
for appellation in listRefApp:
    listJobOfferAssociated = []
    for i in range(nbOffre):
        if listApp[i] == appellation:
            if listOffreId[i] in dictOffreConvert:
                listJobOfferAssociated.append(dictOffreConvert[listOffreId[i]])
            
    setUsers = set(list(df_utility.loc[df_utility['JOBOFFER_ID'].isin(listJobOfferAssociated)]['INDIV_ID']))
    dictRome[appellation] = setUsers
    #print "For App %s, number of application %i" % (str(appellation),len(setUsers))

print "Phase 1 OK"

dict_r = {}
# Now let's create the adjacency matrix
for appellation1 in listRefApp:
    for appellation2 in listRefApp:
        if appellation1 == appellation2:
            # In the diagonal, we store the number of user that have applied to the job
            #dict_r[rome1,rome2] = len(set(dictRome[rome1]))
            appellation1 = appellation2
        else:
            setuser1 = dictRome[appellation1]
            setuser2 = dictRome[appellation2]
            nbuserIntersect = len(setuser1.intersection(setuser2))
            nbuser1 = len(setuser1)
            nbuser2 = len(setuser2)
            sumuser = nbuser1 + nbuser2
            if sumuser > 0:
                r = 2*100*nbuserIntersect/float(sumuser)
                if r > 20:
                    dict_r[appellation1,appellation2] = r
      
print "Phase 2 OK"

with open('../input/affinity_Appellation_matrix_21.csv', 'w') as outfile:
    for apps, r in dict_r.iteritems():
        outfile.write(str(apps[0]))
        outfile.write(",")
        outfile.write(str(apps[1]))
        outfile.write(",")
        outfile.write(("%1.0f"%r))
        outfile.write("\n")
    
# Algo:
# Pour chaque code App, générer une liste de user associé (via un dictionnaire)
# Pour la génération de l'adjacency matrix:
# Pour chaque cell code App 1 / code App 2 => Lancer une jaccard similarity, 
# ou quelque chose d'approchant. Et on a la valeur de la celle en divisant par le len
# de chaque liste associé à chaque code App
    
    
