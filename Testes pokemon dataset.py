import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import os
import sys
import pickle
import cv2
from tqdm import tqdm
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from keras.models import load_model

class_names = ['Abra', 'Alakazam', 'Blastoise', 'Bulbasaur', 'Charizard', 'Charmander', 'Charmeleon', 'Gastly', 
               'Gengar', 'Haunter', 'Ivysaur', 'Kadabra', 'Pikachu', 'Raichu', 'Snorlax', 'Squirtle', 'Venusaur', 'Wartortle']

def get_dataset(path):
    pass

datadir_train = "dataset\\train\\"
datadir_test = "dataset\\test\\"

new_array =[]
training_data = []
testing_data = []
IMG_SIZE = 100

#For train data
for category in tqdm(class_names):
    path = os.path.join(datadir_train, category) # path for each of the pokemons
    class_num = class_names.index(category)
    
    # now we get each of them images
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
        new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        training_data.append([new_array, class_num])
        #plt.imshow(new_array, cmap='gray')
        #plt.show()

#For test data
for category in tqdm(class_names):
    path = os.path.join(datadir_test, category) # path for each of the pokemons
    class_num = class_names.index(category)
    
    # now we get each of them images
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
        new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        training_data.append([new_array, class_num])
        #plt.imshow(new_array, cmap='gray')
        #plt.show()

#For train data   
X_train = []
y_train = []

#For test data
X_test = []
y_test = []

#For train data
for features, label in training_data:
    X_train.append(features)
    y_train.append(label)

X_train = np.array(X_train).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y_train = np.array(y_train)

#For test data
for features, label in testing_data:
    X_test.append(features)
    y_test.append(label)

X_test = np.array(X_test).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y_test = np.array(y_test)

#Dump for train data
pickle_out = open("X_train.pickle", "wb")
pickle.dump(X_train, pickle_out)
pickle_out.close()

pickle_out = open("y_train.pickle", "wb")
pickle.dump(y_train, pickle_out)
pickle_out.close()

#Open for train data
pickle_in = open("X_train.pickle", "rb")
X_train = pickle.load(pickle_in)

pickle_in = open("y_train.pickle", "rb")
y_train = pickle.load(pickle_in)

#Dump for test data
pickle_out = open("X_test.pickle", "wb")
pickle.dump(X_test, pickle_out)
pickle_out.close()

pickle_out = open("y_test.pickle", "wb")
pickle.dump(y_test, pickle_out)
pickle_out.close()

#Open for test data
pickle_in = open("X_test.pickle", "rb")
X_test = pickle.load(pickle_in)

pickle_in = open("y_test.pickle", "rb")
y_test = pickle.load(pickle_in)

## feed to CNN
    
X_train = X_train/255.0
X_test = X_test/255.0

model = Sequential()

model.add(Conv2D(256, (3, 3), input_shape=(X_train.shape[1:])))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

model.add(Dense(64))

model.add(Dense(32))
model.add(Activation('sigmoid'))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=64, epochs=10, validation_split=0.3)

#For save model
model.save('modelPokemon.h5')
print('Model Saved!')

#For save model weights
model.save_weights('modelPokemonWeights.h5')
print('Model Weights Saved!')
