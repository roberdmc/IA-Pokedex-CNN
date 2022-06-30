import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import os
import sys
import pickle
import cv2
from tqdm import tqdm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from keras.models import load_model

#Define o nome das classes
class_names = ['Arbok', 'Arcanine', 'Bellsprout', 'Blastoise', 'Butterfree', 'Charizard', 'Ditto', 'Gengar', 'Jigglypuff',  
               'Machamp', 'Mankey', 'Metapod', 'Mewtwo', 'Parasect', 'Pikachu', 'Poliwag', 'Psyduck', 'Voltorb']
              
def get_dataset(path):
    pass

#Define o diret�rio do conjunto de treinamento
datadir_train = "dataset\\train_17\\"

new_array =[]
training_data = []

#Define o tamanho em altura e largura para o redimensiomanento das imagens
IMG_SIZE = 75

#Gera o training_data para o conjunto de treinamento
for category in tqdm(class_names):
    path = os.path.join(datadir_train, category) #Caminho para cada pok�mon
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

'''
#Exporta os dados de treino (imagens e labels)
pickle_out = open("exported_files\\X_train.pickle", "wb")
pickle.dump(X_train, pickle_out)
pickle_out.close()

pickle_out = open("exported_files\\y_train.pickle", "wb")
pickle.dump(y_train, pickle_out)
pickle_out.close()
'''

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

# Converte feature maps 3D para feature vectors 1D
model.add(Flatten())  

#Hidden Layer
model.add(Dense(512, activation = 'relu'))

#Output Layer
model.add(Dense(len(class_names)))
model.add(Activation('sigmoid'))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=128, epochs=100, validation_split=0.3)

#Salva o modelo gerado e treinado
model.save('exported_files\\modelPokemon.h5')
print('Model Saved!')
