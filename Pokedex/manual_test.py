import cv2, numpy as np
from matplotlib import pyplot as plt

def manual_test(class_names, model, IMG_SIZE):
    #Define o diretório do conjunto de testes
    img = input("File name: ")

    new_array = []
    testing_data = []

    #Cria os vetores para as imagens "X" e labels "y"
    X_test = []
    y_test = []

    img_array = cv2.imread(img, cv2.IMREAD_COLOR)
    new_array = cv2.resize(img_array[:,:,::-1], (IMG_SIZE, IMG_SIZE))
    testing_data.append([new_array, 0])

    #Define os conjuntos de imagens "X" e labels "y"
    for features, label in testing_data:
        X_test.append(features)
        y_test.append(label)

    #Converte os vetores de imagens e labels para np array
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    #Realiza as predições para o conjunto de teste
    predictions = model.predict(X_test)

    #Função para a plotagem das imagens com os seus dados previstos, verdadeiros e a acurácia
    def plot_image(i, predictions_array, img):
        predictions_array, img = predictions_array[i], img[i]
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])

        plt.imshow(img, cmap=plt.cm.binary)

        predicted_label = np.argmax(predictions_array)
        
        color = 'blue'

        plt.xlabel("{}".format(class_names[predicted_label]),
                                      color=color)

    #Plota uma unica imagem
    i = 0
    plt.figure()
    plt.subplot(1,2,1)
    plot_image(i, predictions, X_test)
    plt.show()