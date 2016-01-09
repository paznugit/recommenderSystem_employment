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
listApp = list(df_offre['dc_appelationrome_id'])
nbOffre = len(listOffreId)

#dictOffre = dict(zip(listOffreId,listRome))

listRefApp = list(pd.unique(df_offre['dc_rome_id'].values))

dictApp = {}

# For each 'code Rome' we retrieve the list of users that that applied to
# a job offer link to this 'code Rome'
for appellation in listRefApp:
    listJobOfferAssociated = []
    for i in range(nbOffre):
        if listApp[i] == appellation:
            listJobOfferAssociated.append(listOffreId[i])
            
    listUsers = list(df_utility.loc[df_utility['JOBOFFER_ID'].isin(listJobOfferAssociated)]['INDIV_ID'])
    dictApp[appellation] = listUsers
    print "For Appellation %s, number of users %i" % (appellation,len(listUsers))
   
dict_r = {}
# Now let's create the adjacency matrix
for app1 in listRefApp:
    for app2 in listRefApp:
        if app1 == app1:
            # In the diagonal, we store the number of user that have applied to the job
            #dict_r[app1,app2] = len(set(dictApp[app1]))
            app1 = app1
        else:
            setuser1 = set(dictApp[app1])
            setuser2 = set(dictApp[app2])
            nbuserIntersect = len(setuser1.intersection(setuser2))
            nbuser1 = len(setuser1)
            nbuser2 = len(setuser2)
            sumuser = nbuser1 + nbuser2
            if sumuser > 0:
                r = 2*100*nbuserIntersect/float(sumuser)
                if r > 0:
                    dict_r[app1,app2] = r
                
for app1, r in dict_r.iteritems():
    print app1+","+("%1.0f"%r)
    
# Algo:
# Pour chaque code ROME, générer une liste de user associé (via un dictionnaire)
# Pour la génération de l'adjacency matrix:
# Pour chaque cell code ROME 1 / code ROME 2 => Lancer une jaccard similarity, 
# ou quelque chose d'approchant. Et on a la valeur de la celle en divisant par le len
# de chaque liste associé à chaque code ROME
