import keras
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

# Shift + Alt + E = Sucesso

previsores = pd.read_csv('Breast Cancer Dataset/entradas_breast.csv')
classe = pd.read_csv('Breast Cancer Dataset/saidas_breast.csv')


def criarRede():
    classificador = Sequential()
    #  1ª Camada - Camada com 30 neurônios = camada de atributos
    classificador.add(Dense(units=16, activation='relu', kernel_initializer='random_uniform', input_dim=30))
    #  Avoid overfitting, zering 20% of entring
    classificador.add(Dropout(0.2))
    #  2ª Camada Oculta
    classificador.add(Dense(units=16, activation='relu', kernel_initializer='random_uniform'))
    #  Avoid overfitting, zering 20% of hiden layer
    classificador.add(Dropout(0.2))
    #  Camada de saída
    classificador.add(Dense(units=1, activation='sigmoid'))

    otimizador = keras.optimizers.Adam(lr=0.001, decay=0.0001, clipvalue=0.5)

    classificador.compile(optimizer=otimizador, loss='binary_crossentropy', metrics=['binary_accuracy'])

    return classificador


classificador = KerasClassifier(build_fn=criarRede, epochs=100, batch_size=10)
#  cv=10 === 10-fold cross
resultados = cross_val_score(estimator=classificador, X=previsores, y=classe, cv=10, scoring='accuracy')

media = resultados.mean()
desvio = resultados.std()
