import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

previsores = pd.read_csv('Breast Cancer Dataset/entradas_breast.csv')
classe = pd.read_csv('Breast Cancer Dataset/saidas_breast.csv')


#  Lista de otimizadores, loss function, funcoes de ativção e neuronios para testar
def criarRede(optimizer, loos, kernel_initializer, activation, neurons):
    classificador = Sequential()
    #  1ª Camada - Camada com 30 neurônios = camada de atributos
    classificador.add(Dense(units=neurons, activation=activation, kernel_initializer=kernel_initializer, input_dim=30))
    #  Avoid overfitting, zering 20% of entring
    classificador.add(Dropout(0.2))
    #  2ª Camada Oculta
    classificador.add(Dense(units=neurons, activation=activation, kernel_initializer=kernel_initializer))
    #  Avoid overfitting, zering 20% of hiden layer
    classificador.add(Dropout(0.2))
    #  Camada de saída
    classificador.add(Dense(units=1, activation='sigmoid'))

    classificador.compile(optimizer=optimizer, loss=loos, metrics=['binary_accuracy'])

    return classificador


classificador = KerasClassifier(build_fn=criarRede)
parametros = {'batch_size': [10, 30],
              'epochs': [50, 100],  # se for maior, tende a demorar mais
              'optimizer': ['adam', 'sgd'],
              'loos': ['binary_crossentropy', 'hinge'],
              'kernel_initializer': ['random_uniform', 'normal'],
              'activation': ['relu', 'tanh'],
              'neurons': [16, 8]}  # se for maior, tende a demorar mais
                                                                                        # cross validation
grid_search = GridSearchCV(estimator=classificador, param_grid=parametros, scoring='accuracy', cv=5)
grid_search = grid_search.fit(previsores, classe)

melhores_parametros = grid_search.best_params_
melhor_precisao = grid_search.best_score_

