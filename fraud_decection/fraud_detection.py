# Import the libraries 
import pandas as pd
import numpy as np
import keras

data = pd.read_csv('creditcard.csv')

# data exploration 
data.head()

#data preprocessing 
from sklearn.preprocessing import StandardScaler
data['normalizedAmount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))
data = data.drop(['Amount'], axis = 1)
data.head()

data = data.drop(['Time'], axis = 1)

X = data.iloc[:, data.columns != 'Class']
y = data.iloc[:, data.columns == 'Class']

y.head()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3,
                                                    random_state = 0)
X_train.shape
X_test.shape

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Deep Learning Network
model = Sequential([
        Dense(units = 16, input_dim = 29, activation = 'relu'),
        Dense(units = 24, activation = 'relu'),
        Dropout(0.5),
        Dense(20, activation = 'relu'),
        Dense(24, activation = 'relu'),
        Dense(1, activation = 'sigmoid'),
        ])

model.summary()

# Training the Model 

model.compile(optimizer = 'adam', loss = 'binary_crossentropy',
              metrics = ['accuracy'])
model.fit(X_train, y_train, batch_size = 15, epochs = 5)

score = model.evaluate(X_test, y_test)
print(score)

import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix
from sklearn import svm, datasets


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
        
    print(cm)
    
    plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation = 45)
    plt.yticks(tick_marks, classes)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment = 'center',
                 color = 'white' if cm[i, j] > thresh else 'black')
        
    plt.ylabel('True Label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

y_pred = model.predict(X_test)
y_test = pd.DataFrame(y_test)

cnf_matrix = confusion_matrix(y_test, y_pred.round())

print(cnf_matrix)

plot_confusion_matrix(cnf_matrix, classes=[0, 1])
plt.show()

#
y_pred = model.predict(X)
y_expected = pd.DataFrame(y)
cnf_matrix = confusion_matrix(y_expected, y_pred.round())
plot_confusion_matrix(cnf_matrix, classes=[0, 1])
plt.show()