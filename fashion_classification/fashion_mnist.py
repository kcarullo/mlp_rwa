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

### Training the model 
X_train = training[:, 1:]/255
y_train = training[:, 0]

X_test = testing[:, 1:]/255
y_test = testing[:, 0]

from sklearn.model_selection import train_test_split
X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train,
                                                    test_size = 0.2,
                                                    random_state = 12345)
# reshape the arrays to 28x28x1
X_train = X_train.reshape(X_train.shape[0], *(28,28,1))
X_test = X_test.reshape(X_test.shape[0], *(28,28,1))
X_validate = X_validate.reshape(X_validate.shape[0], *(28,28,1))

X_train.shape # 48000 samples that are 28x28x1
X_test.shape # 10000 samples that are 28x28x1
X_validate.shape # 12000 samples that are 28x28x1

import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras.callbacks import TensorBoard

cnn_model = Sequential()
cnn_model.add(Conv2D(32,3,3, input_shape = (28,28,1), activation = 'relu'))

cnn_model.add(MaxPooling2D(pool_size = (2,2)))

#flatten the model 
cnn_model.add(Flatten())
# add the Dense function
cnn_model.add(Dense(units = 32, activation = 'relu'))
cnn_model.add(Dense(units = 10, activation = 'sigmoid'))

cnn_model.compile(loss = 'sparse_categorical_crossentropy', 
                 optimizer = Adam(lr=0.001), metrics = ['accuracy'])

epochs = 50
cnn_model.fit(X_train, y_train, batch_size = 512, nb_epoch = epochs,
              verbose = 1, validation_data = (X_validate, y_validate))

# Evaluate the Model 
evaluation = cnn_model.evaluate(X_test, y_test)
print('Test Accuracy : {:.3f}'.format(evaluation[1]))

predicted_classes = cnn_model.predict_classes(X_test)
predicted_classes

#create a 5x5 grid
L = 5
W = 5
fig, axes = plt.subplots(L, W, figsize = (12,12))
axes = axes.ravel()

for i in np.arange(0, L * W):
    axes[i].imshow(X_test[i].reshape(28,28))
    axes[i].set_title("Prediction Class = {:0.1f}\n True Class = {:0.1f}".format(predicted_classes[i], y_test[i]))
    axes[i].axis('off')
    
plt.subplots_adjust(wspace = 0.5)

# Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, predicted_classes)
plt.figure(figsize = (14,10))
sns.heatmap(cm, annot=True)

# classification report to view precision, recall, f1-score, and support 
from sklearn.metrics import classification_report

num_classes = 10
target_names = ["Class {}".format(i) for i in range(num_classes)]

print(classification_report(y_test, predicted_classes, target_names = target_names))

"""
0 t-shirt/top
1 trouser
2 pullover
3 dress
4 coat
5 sandal
6 shirt
7 sneaker
8 bag
9 ankle boot
"""