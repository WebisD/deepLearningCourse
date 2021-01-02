import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential  # Modelo Sequencial: uma camada após a outra
from keras.layers import Dense  # Fully conected: um neuronio conectado a todos os outros
from sklearn.preprocessing import LabelEncoder  # String to number
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix
import numpy as np
# Shift + Alt + E = Sucesso

base = pd.read_csv('Multiclass Classification/Iris Dataset/iris.csv')

# Função para dividir base de dados.
                    #Todas as linhas / 4 1ª colunas
previsores = base.iloc[:, 0:4].values
classe = base.iloc[:, 4].values

# 0 - Iris-versicolor // 1 - Iris-virginica // 2 - Iris-setosa
# Camada de saida com 3 neuronios -> 001 : Versi // 010 : Virgin // 100 : setosa
labelenconder = LabelEncoder()
classe = labelenconder.fit_transform((classe))

#  Avoid erro, transforming 1 shape  array in 3 shape array
classe_dummy = np_utils.to_categorical(classe)
# 25% for tests
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe_dummy, test_size=0.25)

classificador = Sequential()

# Quantidade de neuronios na camada oculta = (quant. entradas(4) + quant. saidas(3)) / 2
# input_dim = quant. de previsores
# 2 camadas ocultas:
classificador.add(Dense(units=4, activation='relu', input_dim=4))
classificador.add(Dense(units=4, activation='relu'))
# Camada de saída
classificador.add(Dense(units=3, activation='softmax'))

classificador.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

classificador.fit(previsores_treinamento, classe_treinamento, batch_size=10, epochs=1000)

#  Avaliação automaticaa
#  [0] loss function [1] % de acerto
resultado = classificador.evaluate(previsores_teste, classe_teste)
# probabilidade para cada tipo de saida
previsoes = classificador.predict(previsores_teste)
previsoes = (previsoes > 0.5)
# array com indices com maiores valores
classe_teste_2 = [np.argmax(t) for t in classe_teste]
previsoes_2 = [np.argmax(t) for t in previsoes]

#  Matriz de confusão :P
matriz = confusion_matrix(previsoes_2, classe_teste_2)
