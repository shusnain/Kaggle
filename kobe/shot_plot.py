# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 01:15:40 2016

@author: S_Husnain
"""

import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame
import pandas as pd

fig = plt.figure()
ax1 = fig.add_subplot(111)

## the data
path = "D:/Dropbox/Data Science/Kaggle/kobe"
data = pd.read_csv(path + '/data.csv')
frame = DataFrame(data)
cframe = frame[frame.shot_made_flag.notnull()]
cframe.loc[(cframe['shot_made_flag'] == 1) , 'loc_x']
x_make = cframe.loc[(cframe['shot_made_flag'] == 1) , 'loc_x']
y_make = cframe.loc[(cframe['shot_made_flag'] == 1) , 'loc_y']
x_miss = cframe.loc[(cframe['shot_made_flag'] == 0) , 'loc_x']
y_miss = cframe.loc[(cframe['shot_made_flag'] == 0) , 'loc_y']
print(len(x_make)/(len(x_make)+len(x_miss)))

## left panel
ax1.scatter(x_make,y_make,color='blue',s=20,edgecolor='none',alpha =0.5)
ax1.scatter(x_miss,y_miss,color='red',s=20,edgecolor='none', alpha =0.5)
ax1.set_aspect(1./ax1.get_data_ratio()) # make axes square
fig.set_size_inches(56.4, 30)
plt.savefig(path + '/shots.png', dpi = 100)