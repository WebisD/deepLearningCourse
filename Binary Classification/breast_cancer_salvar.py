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

classificador_json = classificador.to_json()
# Guarda a estrutura da rede neural(neuronios, camdas, funca ativação)
with open('classificador_breast.json', 'w') as json_file:
    json_file.write(classificador_json)

# Salva os pesos da rede
classificador.save_weights('classificador_breast.h5')
