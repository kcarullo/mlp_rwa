# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 12:23:42 2019

@author: Kevin
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn

dataset = pd.read_csv('financial_data.csv')

# EDA 

dataset.head()
dataset.columns
dataset.describe()

# Cleaning the data 

# Removing NaN
dataset.isna().any() # no NAs

# Histograms 

dataset2 = dataset.drop(columns = ['entry_id', 'pay_schedule', 'e_signed'])

fig = plt.figure(figsize=(15,12))
plt.suptitle('Histograms of Numerical Columns', fontsize=20)
for i in range(dataset2.shape[1]):
    plt.subplot(6,3,i+1)
    f = plt.gca()
    f.set_title(dataset2.columns.values[i])
    
    vals = np.size(dataset2.iloc[:,i].unique())
    # Limits number of bins to 100 if there are more then 100 unique numbers
    if vals >= 100:
        vals = 100
        
    plt.hist(dataset2.iloc[:,i], bins=vals, color='#3F5D7D')
plt.tight_layout(rect=[0,0.03,1,0.95])

# Correlation with response variable (Note: Models like RF are not linear like these)

dataset2.corrwith(dataset.e_signed).plot.bar(
        figsize= (15,15), title = "Correlation with E Signed", fontsize = 10,
        rot = 30, grid = True)

# Correlation Matrix
sn.set(style="white")

# Compute the correlation matrix
corr = dataset2.corr()

# Generate a mask for the upper triangle 
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure 

f, ax = plt.subplots(figsize=(18,15))

#Generate a custom diverging colormap 
cmap = sn.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sn.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
           square=True, linewidths=.5, cbar_kws={"shrink": .5})
