# -*- coding: utf-8 -*-
"""
Created on Sat Jan 09 23:17:36 2016

@author: Guillaume
"""

import numpy as np

a = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
q = 90
r = np.percentile(a,q)
print r