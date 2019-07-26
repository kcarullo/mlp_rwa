# Import the libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from dateutil import parser

# Load and view the dataset
dataset = pd.read_csv('appdata10.csv')
dataset.head()
dataset.describe()

# Data Cleaning 

dataset["hour"] = dataset.hour.str.slice(1, 3).astype(int)

# Plotting 
dataset2 = dataset.copy().drop(columns = ['user', 'screen_list', 'enrolled_date', 'first_open', 'enrolled'])
dataset2.head()

# Histograms 
plt.suptitle('Histograms of Numberical Columns', fontsize = 20)
for i in range(1, dataset2.shape[1] + 1):
    plt.subplot(3, 3, i)
    f = plt.gca()
    f.set_title(dataset2.columns.values[i -1])
    
    vals = np.size(dataset2.iloc[:, i -1].unique())
    
    plt.hist(dataset2.iloc[:, i - 1], bins = vals, color = '#3F5D7D')
             
# Correlation with Response 
dataset2.corrwith(dataset.enrolled).plot.bar(figsize = (20, 10),
                  title = 'Correlation with Response Variable', 
                  fontsize = 15, rot = 45,
                  grid = True)

# Correlation Matrix 
sn.set(style='white', font_scale=2)

# Compute the correlation matrix 
corr = dataset2.corr()

# Generate a mask for the upper triangle 
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure 
f, ax = plt.subplots(figsize=(18, 15))
f.suptitle("Correlation Matrix", fontsize = 30)

# Generate a custom diverging colormap
cmap = sn.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with thte mask and correct aspect ratio
sn.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
           square=True, linewidths=.5, cbar_kws={"shrink": .5})

# Feature Engineering 



