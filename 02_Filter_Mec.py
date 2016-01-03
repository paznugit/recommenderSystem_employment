# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 09:27:56 2015

@author: Guillaume
"""

import pandas as pd
from random import shuffle
import pickle
import numpy as np

csv_input = '../input/dm_mec.csv'
csv_cible = '../input/dm_mec_ng.csv'

#==============================================================================
# Convert a file which represent:
# - For each line a link between a job offer and a person (with the score associated to it)
#
# To a new file which represent:
# - The same information but without the individual who candidated less than x times
# - The converted identifier fo person and job offer (take integer from 0 to max number of distinct elements)
#==============================================================================

# Number minimum of candidature a person should have to be in the dataset
candidature_threshold = 5

# Loading the dataframe
dataframe = pd.read_csv(csv_input, sep = '\t', names = ['KC_OFFRE_ID','DN_INDIVIDU_NATIONAL','SCORE'])

print "##################"
print "Before filtering:"
print "nb individu: %i" % len(pd.unique(dataframe['DN_INDIVIDU_NATIONAL'].values))
print "nb offre: %i" % len(pd.unique(dataframe['KC_OFFRE_ID'].values))

# Counting per individual the number of links
count_mec = dataframe.groupby('DN_INDIVIDU_NATIONAL')['KC_OFFRE_ID'].count().reset_index()
# Filter to get only the individuals with at least x links
list_indiv = list(count_mec.loc[count_mec['KC_OFFRE_ID'] >= 5]['DN_INDIVIDU_NATIONAL'])
dataframe = dataframe.loc[dataframe['DN_INDIVIDU_NATIONAL'].isin(list_indiv)]

# Get the list of individu and produce a conversion index dictionnary
listIndividu = pd.unique(dataframe['DN_INDIVIDU_NATIONAL'].values)
shuffle(listIndividu)
listIndex = range(len(listIndividu))
dictIndiv = dict(zip(listIndividu,listIndex))

# Get the list of job offers and produce a conversion index dictionnary
listOffre = pd.unique(dataframe['KC_OFFRE_ID'].values)
shuffle(listOffre)
listIndex = range(len(listOffre))
dictOffre = dict(zip(listOffre,listIndex))

# We convert indexes
indiv_id = list(dataframe.apply(lambda x:  dictIndiv[x['DN_INDIVIDU_NATIONAL']],axis=1))
job_id = list(dataframe.apply(lambda x: dictOffre[x['KC_OFFRE_ID']],axis=1))
score = list(dataframe['SCORE'])
print len(job_id)
print len(np.unique(job_id))
dataframe = pd.DataFrame(data = {'INDIV_ID': indiv_id, 'JOBOFFER_ID': job_id, 'SCORE': score})
#dataframe = dataframe.drop(['DN_INDIVIDU_NATIONAL','KC_OFFRE_ID'], axis=1)

# We save the indiv dictionnary
outputIndiv = open('../input/indiv_dict.pkl', 'wb')
pickle.dump(dictIndiv, outputIndiv)
outputIndiv.close()
with open('../input/indiv_dict.csv', 'w') as outfile:
    for key,value in dictIndiv.iteritems():
        outfile.write("%i"%key)
        outfile.write(",")
        outfile.write(str(value))
        outfile.write("\n")
   
# We save the job offer dictionnary     
outputJob = open('../input/joboffer_dict.pkl', 'wb')
pickle.dump(dictOffre, outputJob)
outputJob.close()
with open('../input/joboffer_dict.csv', 'w') as outfile:
    for key,value in dictOffre.iteritems():
        outfile.write(str(key))
        outfile.write(",")
        outfile.write(str(value))
        outfile.write("\n")
    
'''pkl_file = open('myfile.pkl', 'rb')
mydict2 = pickle.load(pkl_file)
pkl_file.close()'''

print len(pd.unique(dataframe['JOBOFFER_ID'].values))
# Remove the indiv_id-joboffer duplicate of the dataset, since they exist
dataframe.drop_duplicates(subset=['INDIV_ID','JOBOFFER_ID'], inplace = True)

print "##################"
print "After filtering:"
print "nb individu: %i" % len(pd.unique(dataframe['INDIV_ID'].values))
print "nb offre: %i" % len(pd.unique(dataframe['JOBOFFER_ID'].values))

# Save to file
dataframe.to_csv(csv_cible, index=False, float_format='%i')





