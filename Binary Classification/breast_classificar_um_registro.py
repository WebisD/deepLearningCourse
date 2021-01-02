import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout

previsores = pd.read_csv('Breast Cancer Dataset/entradas_breast.csv')
classe = pd.read_csv('Breast Cancer Dataset/saidas_breast.csv')

classificador = Sequential()
#  1ª Camada - Camada com 30 neurônios = camada de atributos
classificador.add(Dense(units=16, activation='relu', kernel_initializer='normal', input_dim=30))
#  Avoid overfitting, zering 20% of entring
classificador.add(Dropout(0.2))
#  2ª Camada Oculta
classificador.add(Dense(units=16, activation='relu', kernel_initializer='normal'))
#  Avoid overfitting, zering 20% of hiden layer
classificador.add(Dropout(0.2))
#  Camada de saída
classificador.add(Dense(units=1, activation='sigmoid'))

classificador.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])

classificador.fit(previsores, classe, batch_size=10, epochs=100)
#  radius_mean, texture,mean,
novo = np.array([[15.80, 8.34, 118, 900, 0.10, 0.26, 0.08, 0.134, 0.178, 0.20, 0.05, 1098, 0.87, 4500, 145.2, 0.005, 0.04, 0.05, 0.015, 0.03, 0.007, 23.15, 16.64, 178.5, 2018, 0.14, 0.185, 0.84, 158, 0.363]])

#~= 1 maligno else beligno
previsao = classificador.predict(novo)
previsao = (previsao > 0.5)

print("Maligno?: ", previsao)
