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

class_names = ['Arbok', 'Arcanine', 'Blastoise', 'Butterfree', 'Charizard', 'Gengar', 'Gyarados', 'Jigglypuff',  
               'Machamp', 'Mewtwo', 'Ninetales', 'Pikachu', 'Psyduck', 'Starmie', 'Tauros', 'Vileplume', 'Voltorb']
              
def get_dataset(path):
    pass

datadir_train = "dataset\\train_17\\"

new_array =[]
training_data = []
IMG_SIZE = 75

#For train data
for category in tqdm(class_names):
    path = os.path.join(datadir_train, category) # path for each of the pokemons
    class_num = class_names.index(category)
    
    # now we get each of them images
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
        new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        training_data.append([new_array, class_num])
        #plt.imshow(new_array, cmap='gray')
        #plt.show()

#For train data   
X_train = []
y_train = []

#For train data
for features, label in training_data:
    X_train.append(features)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

#Dump for train data
pickle_out = open("X_train.pickle", "wb")
pickle.dump(X_train, pickle_out)
pickle_out.close()

pickle_out = open("y_train.pickle", "wb")
pickle.dump(y_train, pickle_out)
pickle_out.close()

## feed to CNN

model = Sequential()

model.add(Conv2D(256, (3, 3), input_shape=(IMG_SIZE, IMG_SIZE, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

model.add(Dense(64))

model.add(Dense(25))
model.add(Activation('sigmoid'))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=64, epochs=12, validation_split=0.3)

#For save model
model.save('modelPokemon.h5')
print('Model Saved!')
