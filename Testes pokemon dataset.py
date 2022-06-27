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

class_names = ['Abra', 'Alakazam', 'Blastoise', 'Bulbasaur', 'Charizard', 'Charmander', 'Charmeleon', 'Gastly', 
               'Gengar', 'Haunter', 'Ivysaur', 'Kadabra', 'Pikachu', 'Raichu', 'Snorlax', 'Squirtle', 'Venusaur', 'Wartortle']

def get_dataset(path):
    pass

print('test')
print('oi')
datadir = "C:\\Users\\Roberval.junior\\Downloads\\tensorflow-course-master\\tensorflow-course-master\\dataset\\train\\"

new_array =[]
training_data = []
IMG_SIZE = 100


for category in tqdm(class_names):
    path = os.path.join(datadir, category) # path for each of the pokemons
    class_num = class_names.index(category)
    
    # now we get each of them images
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
        new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        training_data.append([new_array, class_num])
        #plt.imshow(new_array, cmap='gray')
        #plt.show()

   
X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

#dump

pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()



pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle", "rb")
y = pickle.load(pickle_in)

## feed to CNN
    
X = X/255.0

model = Sequential()

model.add(Conv2D(256, (3, 3), input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

model.add(Dense(64))

model.add(Dense(17))
model.add(Activation('sigmoid'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X, y, batch_size=64, epochs=5, validation_split=0.3)

