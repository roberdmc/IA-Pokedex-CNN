import tensorflow as tf
import numpy as np
import os
import sys
import pickle
import cv2
from tqdm import tqdm
from matplotlib import pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

#Define o nome das classes
class_names = ['Abra', 'Bulbasaur', 'Charmander', 'Gastly', 'Meowth', 'Pidgey', 'Pikachu', 'Squirtle', 'Staryu']
              
def get_dataset(path):
    pass

#Define o diretório do conjunto de treinamento
datadir_train = "dataset_new\\train_17\\"

new_array =[]
training_data = []

#Define o tamanho em altura e largura para o redimensiomanento das imagens
IMG_SIZE = 100

#Gera o training_data para o conjunto de treinamento
for category in tqdm(class_names):
    path = os.path.join(datadir_train, category) #Caminho para cada pokémon
    class_num = class_names.index(category)
    
    #Processa cada uma das imagens
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
        new_array = cv2.resize(img_array[:,:,::-1], (IMG_SIZE, IMG_SIZE))
        training_data.append([new_array, class_num])


#Cria os vetores para as imagens "X" e labels "y"  
X_train = []
y_train = []

#Define os conjuntos de imagens "X" e labels "y"
for features, label in training_data:
    X_train.append(features)
    y_train.append(label)

#Converte os vetores de imagens e labels para np array
X_train = np.array(X_train)
y_train = np.array(y_train)

##Processamento CNN
model = Sequential()

#Input Layer
model.add(Conv2D(32, (3, 3), activation = 'relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(256, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(512, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Converte feature maps 3D para feature vectors 1D
model.add(Flatten())  

#Hidden Layer
model.add(Dense(1024, activation = 'relu'))
model.add(Dense(1024, activation = 'relu'))

#Output Layer
model.add(Dense(len(class_names)))
model.add(Activation('sigmoid'))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=128, epochs=20, validation_split=0.3)

#Salva o modelo gerado e treinado
model.save('exported_files\\modelPokemon.h5')
print('Model Saved!')
