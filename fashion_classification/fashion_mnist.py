"""
Fashion training set consists of 70000 images divided into 60k training and 10k
testing samples. Dataset samples consists of 28x28 greyscale image associated with 
a label from 10 classes. 
t-shirt/top, trouser, pullover, dress, coat, sandal, shirt, sneaker, bag, ankle boot

Each image is 28 pixels in height and width, for a total of 784 pixels in total. 
Each pixel has a single pixel-value associated with it, indicating the lightness 
or darkness of that pixel, with higher numbers meaning darker. The pixel-value
is an integer between 0 and 255.

"""
# import the libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import random
fashion_train_df = pd.read_csv('fashion-mnist_train.csv', sep = ',')
fashion_test_df = pd.read_csv('fashion-mnist_test.csv', sep = ',')

#### visualizing the dataset 
fashion_train_df.head()
fashion_train_df.tail()

fashion_train_df.shape
fashion_test_df.shape

# create training and testing arrays  
training = np.array(fashion_train_df, dtype = 'float32')
testing = np.array(fashion_test_df, dtype = 'float32')

i = random.randint(1,60000)
# used 1 because index 0 is row #s
plt.imshow(training[i, 1:].reshape(28,28))
label = training[i, 0]
label

# Let's view more images in a grid format
# Define the dimensions of the plot grid 
W_grid = 15
L_grid = 15

# fig, axes = plt.subplots(L_grid, W_grid)
# subplot return the figure object and axes object 
# we can use the axes object to plot specific figures at various locations 
fig, axes = plt.subplots(L_grid, W_grid, figsize = (17,17))

axes = axes.ravel() # flaten the 15 x 15 maatrix into 225 array 

n_training = len(training) # get the length of the training dataset

# select a rondom number from 0 to n_training 
for i in np.arange(0, W_grid * L_grid): # create evenly spaced variables 
    
    # selct a random number 
    index = np.random.randint(0, n_training)
    # read and display an image with the selected index 
    axes[i].imshow(training[index,1:].reshape((28,28)))
    axes[i].set_title(training[index,0], fontsize = 8)
    axes[i].axis('off')
    
plt.subplots_adjust(hspace = 0.4)

