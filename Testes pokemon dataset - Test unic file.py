import os, sys, pickle, cv2, numpy as np, tensorflow as tf
from tqdm import tqdm
from matplotlib import pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

#Define o nome das classes
class_names = ['Abra', 'Arbok', 'Arcanine', 'Bellsprout', 'Blastoise','Bulbasaur', 'Butterfree',
               'Charizard', 'Charmander', 'Ditto', 'Gastly', 'Gengar', 'Jigglypuff', 'Machamp', 
               'Mankey', 'Meowth', 'Metapod', 'Mewtwo', 'Parasect', 'Pidgey', 'Pikachu', 'Poliwag', 
               'Psyduck','Squirtle', 'Staryu', 'Voltorb']

def get_dataset(path):
    pass

#Carrega a rede neural gerada no aprendizado
model = load_model('exported_files\\modelPokemon.h5')
print('Model Loaded!')
model.summary()

#Define o diret�rio do conjunto de testes
datadir_test = "dataset\\test_17_dupla\\"

new_array =[]
testing_data = []

#Define o tamanho em altura e largura para o redimensiomanento das imagens
IMG_SIZE = 50

#Gera o testing_data para o conjunto de testes
for category in tqdm(class_names):
    path = os.path.join(datadir_test, category) #Caminho para cada pok�mon
    class_num = class_names.index(category)
    
    #Processa cada uma das imagens
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
        new_array = cv2.resize(img_array[:,:,::-1], (IMG_SIZE, IMG_SIZE))
        testing_data.append([new_array, class_num])

#Cria os vetores para as imagens "X" e labels "y"
X_test = []
y_test = []

#Define os conjuntos de imagens "X" e labels "y"
for features, label in testing_data:
    X_test.append(features)
    y_test.append(label)

#Converte os vetores de imagens e labels para np array
X_test = np.array(X_test)
y_test = np.array(y_test)

#Calcula a acur�cia para o conjunto de testes
test_loss, test_acc = model.evaluate(X_test,  y_test, batch_size=64, verbose=2)
print('\nTest accuracy:', test_acc)

#Realiza as predi��es para o conjunto de teste
predictions = model.predict(X_test)
predictions[0]
np.argmax(predictions[0])
y_test[0]

#Fun��o para a plotagem das imagens com os seus dados previstos, verdadeiros e a acur�cia
def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
      color = 'blue'
    else:
      color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                  100*np.max(predictions_array),
                                  class_names[true_label]),
                                  color=color)

#Plota as primeiras imagens de teste, as labels preditas, as labels verdadeiras e a acur�cia.
num_rows = 5
num_cols = 6
#num_images = 12
num_images = len(class_names)

plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, y_test, X_test)
plt.show()

plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i+num_images, predictions, y_test, X_test)
plt.show()
