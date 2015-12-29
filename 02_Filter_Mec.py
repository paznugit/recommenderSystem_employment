# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 09:27:56 2015

@author: Guillaume
"""

import pandas as pd
from random import shuffle
import pickle

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
dataframe = pd.read_csv(csv_input, sep = '\t')



# Get the list of individu and produce a conversion index dictionnary
listIndividu = pd.unique(dataframe['DN_INDIVIDU_NATIONAL'].values)
shuffle(listIndividu)
nbindiv = len(listIndividu)
listIndex = range(nbindiv)
dictIndiv = dict(zip(listIndividu,listIndex))
print "nb individu: %i" % nbindiv

# Get the list of job offers and produce a conversion index dictionnary
listOffre = pd.unique(dataframe['KC_OFFRE_ID'].values)
shuffle(listOffre)
nbOffre = len(listOffre)
listIndex = range(len(listOffre))
dictOffre = dict(zip(listOffre,listIndex))
print "nb offre: %i" % nbOffre

#print dataframe
# We convert indexes
indiv_id = list(dataframe.apply(lambda x:  dictIndiv[x['DN_INDIVIDU_NATIONAL']],axis=1))
#print indiv_id
job_id = list(dataframe.apply(lambda x: dictOffre[x['KC_OFFRE_ID']],axis=1))
#print job_id
dataframe = dataframe.join(pd.DataFrame(data = {'INDIV_ID': indiv_id, 'JOBOFFER_ID': job_id}))
dataframe = dataframe.drop(['DN_INDIVIDU_NATIONAL','KC_OFFRE_ID'], axis=1)

outputIndiv = open('../input/indiv_dict.pkl', 'wb')
pickle.dump(dictIndiv, outputIndiv)
outputIndiv.close()
with open('../input/indiv_dict.csv', 'w') as outfile:
    for key,value in dictIndiv.iteritems():
        outfile.write("%i"%key)
        outfile.write(",")
        outfile.write(str(value))
        outfile.write("\n")
        
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

# Counting per individual the number of links
count_mec = dataframe.groupby('INDIV_ID')['JOBOFFER_ID'].count().reset_index()
# Filter to get only the individuals with at least x links
list_indiv = list(count_mec.loc[count_mec['JOBOFFER_ID'] >= 5]['INDIV_ID'])
dataframe = dataframe[dataframe['INDIV_ID'].isin(list_indiv)]

#dataframe = dataframe[:30]

# Save to file
dataframe.to_csv(csv_cible, index=False, float_format='%i')





