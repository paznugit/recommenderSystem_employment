# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 10:42:30 2015

@author: Guillaume
"""

import pandas as pd

csv_input = '../input/dm_mec_21_ng.csv'
csv_cible = '../input/dm_mec_21_ng_bo.csv'

#==============================================================================
# Convert a file which represent:
# - For each line a link between a job offer and a person (with the score associated to it)
#
# To a new file which represent:
# - The same information but with a boolean score (1 if score > 0 (which is always
#   in this sparse matrix), else 0)
#==============================================================================

# Loading the dataframe
dataframe = pd.read_csv(csv_input)

# Change the score
dataframe['SCORE'] = 1

# Save to file
dataframe.to_csv(csv_cible, index=False, float_format='%i')