# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 18:12:53 2016

@author: S_Husnain
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.nonparametric.kde import KDEUnivariate
from statsmodels.nonparametric import smoothers_lowess
from pandas import Series, DataFrame
from patsy import dmatrices
from sklearn import datasets, svm

wdir = 'D:/Dropbox/Data Science/Kaggle/Titanic'
df = pd.read_csv(wdir + '/train.csv')

#cleanup data
df = df.drop(['Ticket','Cabin'], axis=1).dropna()

alpha_level = 1

fig = plt.figure(figsize=(18,4), dpi=1600)
ax1 = fig.add_subplot(141)
survivors_by_class = df.Pclass[df.Survived == 1].value_counts()
survivors_by_class.plot(kind='bar', color='green', alpha = alpha_level)
ax1.set_xlim(-1, len(survivors_by_class))
ax1.set_xticklabels(['High', 'Middle', 'Low'], rotation = 0)
plt.ylabel('People')
plt.title('Survivors by Class')

ax2 = fig.add_subplot(142)
survivor_rate_by_class = df.Pclass[df.Survived == 1].value_counts()/df.Pclass.value_counts()
survivor_rate_by_class.plot(kind='bar', color='blue', alpha = alpha_level)
ax2.set_xlim(-1, len(survivor_rate_by_class))
ax2.set_xticklabels(['High', 'Middle', 'Low'], rotation = 0)
plt.title('Survival rate by Class')

ax3 = fig.add_subplot(143)
survivors_by_class = df.Sex[df.Survived == 1].value_counts()
survivors_by_class.plot(kind='bar', color='green', alpha = alpha_level)
ax3.set_xlim(-1, len(survivors_by_class))
ax3.set_xticklabels(['Female', 'Male'], rotation = 0)
plt.title('Survivors by Sex')

ax4 = fig.add_subplot(144)
survivor_rate_by_class = df.Sex[df.Survived == 1].value_counts()/df.Sex.value_counts()
survivor_rate_by_class.plot(kind='bar', color='blue', alpha = alpha_level)
ax4.set_xlim(-1, len(survivor_rate_by_class))
ax4.set_xticklabels(['Female', 'Male'], rotation = 0)
plt.title('Survival rate by Sex')

#fig.set_xlim(-1, len(survived_by_class))
# # specifies the parameters of our graphs
# fig = plt.figure(figsize=(18,6), dpi=1600) 
# alpha=alpha_scatterplot = 0.2 
# alpha_bar_chart = 0.55

# # lets us plot many diffrent shaped graphs together 
# ax1 = plt.subplot2grid((2,3),(0,0))
# # plots a bar graph of those who surived vs those who did not.               
# df.Survived.value_counts().plot(kind='bar', alpha=alpha_bar_chart)
# # this nicely sets the margins in matplotlib to deal with a recent bug 1.3.1
# ax1.set_xlim(-1, 2)
# # puts a title on our graph
# plt.title("Distribution of Survival, (1 = Survived)")    

# plt.subplot2grid((2,3),(0,1))
# plt.scatter(df.Survived, df.Age, alpha=alpha_scatterplot)
# # sets the y axis lable
# plt.ylabel("Age")
# # formats the grid line style of our graphs                          
# plt.grid(b=True, which='major', axis='y')  
# plt.title("Survival by Age,  (1 = Survived)")

# ax3 = plt.subplot2grid((2,3),(0,2))
# df.Pclass.value_counts().plot(kind="barh", alpha=alpha_bar_chart)
# ax3.set_ylim(-1, len(df.Pclass.value_counts()))
# plt.title("Class Distribution")

# plt.subplot2grid((2,3),(1,0), colspan=2)
# # plots a kernel density estimate of the subset of the 1st class passangers's age
# df.Age[df.Pclass == 1].plot(kind='kde')    
# df.Age[df.Pclass == 2].plot(kind='kde')
# df.Age[df.Pclass == 3].plot(kind='kde')
#  # plots an axis lable
# plt.xlabel("Age")    
# plt.title("Age Distribution within classes")
# # sets our legend for our graph.
# plt.legend(('1st Class', '2nd Class','3rd Class'),loc='best') 

# ax5 = plt.subplot2grid((2,3),(1,2))
# df.Embarked.value_counts().plot(kind='bar', alpha=alpha_bar_chart)
# ax5.set_xlim(-1, len(df.Embarked.value_counts()))
# # specifies the parameters of our graphs
# plt.title("Passengers per boarding location") 