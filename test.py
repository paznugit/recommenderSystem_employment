# -*- coding: utf-8 -*-
"""
Created on Sat Jan 09 23:17:36 2016

@author: Guillaume
"""

import pandas as pd

# Récupérer les codeRome auquel un individu a postulé:
indiv_id = 1970

csv_input = '../input/dm_mec_21_ng_bo.csv'
csv_jobofferdict__input = '../input/joboffer_dict_21.csv'
csv_dmoff_input = '../input/dm_off_21_ng.csv'

df_utility = pd.read_csv(csv_input)
listejobofferid = list(df_utility.loc[df_utility['INDIV_ID'] == indiv_id]['JOBOFFER_ID'])

df_convertJobOffer = pd.read_csv(csv_jobofferdict__input, names = ['KC_OFFRE_ID','JOBOFFER_ID'])
listekcoffre = list(df_convertJobOffer.loc[df_convertJobOffer['JOBOFFER_ID'].isin(listejobofferid)]['KC_OFFRE_ID'])


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
listerome = list(pd.unique(df_offre.loc[df_offre['kc_offre'].isin(listekcoffre)]['dc_rome_id']))
print listerome