# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 09:15:05 2015

@author: IGPL3460
"""

import pandas as pd

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

cols_to_retain = [ 'dc_naturecontrat_id', 'dc_typecontrat_id',
                  'dc_qualification_id', 'dc_langue_1_id', 'dc_permis_1_id']
                
cat_df = df_offre[cols_to_retain]
cat_dict = cat_df.T.to_dict().values()


from sklearn.feature_extraction import DictVectorizer as DV

vectorizer = DV( sparse = False )
vec_x_cat_train = vectorizer.fit_transform( cat_dict )
print vec_x_cat_train[:5]

# Attention qualificationId TODO