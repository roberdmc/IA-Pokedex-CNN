import os, cv2, numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

def pair_test(class_names, model, IMG_SIZE):
    #Define o diretório do conjunto de testes
    datadir_test = "dataset\\test_pair\\"

    new_array =[]
    testing_data = []

    #Gera o testing_data para o conjunto de testes
    for category in tqdm(class_names):
        path = os.path.join(datadir_test, category) #Caminho para cada pokémon
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

    #Calcula a acurácia para o conjunto de testes
    test_loss, test_acc = model.evaluate(X_test,  y_test, batch_size=64, verbose=2)
    print('\nTest accuracy:', test_acc)

    #Realiza as predições para o conjunto de teste
    predictions = model.predict(X_test)

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

        plt.xlabel("{} ({})".format(class_names[predicted_label],
                                    class_names[true_label]),
                                    color=color)

    #Plota as primeiras imagens de teste, as labels preditas, as labels verdadeiras e a acurácia.
    num_rows = 5
    num_cols = 5
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