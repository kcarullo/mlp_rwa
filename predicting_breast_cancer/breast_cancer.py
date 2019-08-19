import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
cancer
cancer.keys()
print(cancer['DESCR'])
print(cancer['target'])
print(cancer['target_names'])
print(cancer['feature_names'])

cancer['data'].shape
df_cancer = pd.DataFrame(np.c_[cancer['data'], cancer['target']], 
                         columns = np.append(cancer['feature_names'], ['target']))

df_cancer.head()
df_cancer.tail()

# Visualizing the data
sns.pairplot(df_cancer, hue = 'target', vars = ['mean radius', 'mean texture',
                                'mean area', 'mean perimeter', 'mean smoothness'])

sns.countplot(df_cancer['target'])

sns.scatterplot(x = 'mean area', y = 'mean smoothness', hue = 'target', data = df_cancer)

plt.figure(figsize = (20,10))
sns.heatmap(df_cancer.corr(), annot = True)
