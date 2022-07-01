from keras.models import load_model
from manual_test import manual_test
from pair_test import pair_test
from total_test import total_test
from train_network import train_network

#Define o nome das classes
class_names = ['Arbok', 'Arcanine', 'Bellsprout', 'Blastoise','Bulbasaur', 'Butterfree',
               'Charizard', 'Charmander', 'Ditto', 'Gastly', 'Gengar', 'Jigglypuff', 'Machamp', 
               'Mankey', 'Meowth', 'Metapod', 'Mewtwo', 'Parasect', 'Pidgey', 'Pikachu', 'Poliwag', 
               'Psyduck', 'Snorlax', 'Squirtle','Voltorb']

#Carrega a rede neural gerada no ultimo aprendizado
model = load_model('exported_files\\modelPokemon.h5')

#Define o tamanho em altura e largura para o redimensiomanento das imagens
IMG_SIZE = 75

menu_options = {
    1: 'Total test',
    2: 'Pair test',
    3: 'Manual test',
    4: 'Train network',
    5: 'Load best model',
    6: 'Exit',
}

def print_menu():
    print('Menu:')
    for key in menu_options.keys():
        print (key, '--', menu_options[key] )

if __name__=='__main__':
    while(True):
        print_menu()
        option = ''
        try:
            option = int(input('Enter your choice: '))
        except:
            print('Wrong input. Please enter a number ...')
        #Check what choice was entered and act accordingly
        if option == 1:
            total_test(class_names, model, IMG_SIZE)
        elif option == 2:
            pair_test(class_names, model, IMG_SIZE)
        elif option == 3:
            try:
                manual_test(class_names, model, IMG_SIZE)
            except:
                print('\nArquivo invalido!\n')
        elif option == 4:
            train_network(class_names, model, IMG_SIZE)
            model = load_model('exported_files\\modelPokemon.h5')
            print('\nNew Model loaded!\n')
        elif option == 5:
            model = load_model('exported_files\\bestModel.h5')
            print('\nBest Model loaded!\n')
        elif option == 6:
            print('Exiting.')
            exit()
        else:
            print('Invalid option. Please enter a number between 1 and 5.')