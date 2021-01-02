import pandas as pd
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
#  Usando camadas densas na rede neural ( um neurônio é ligado a todos os outros da camada subsequente) - fully conected
from keras.layers import Dense
from sklearn.metrics import confusion_matrix, accuracy_score

# Shift + Alt + E = Sucesso

previsores = pd.read_csv("Breast Cancer Dataset/entradas_breast.csv")
classe = pd.read_csv('Breast Cancer Dataset/saidas_breast.csv')

#  75% para treino e 25% para teste
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.25)

classificador = Sequential()

#  ----------------- 1ª Camada Oculta -------------------
#  camada de entrada com 30 neurônios (caracteiristicas como radius_mean, texture_mean, etc.)
classificador.add(Dense(units=16, activation='relu', kernel_initializer='random_uniform', input_dim=30))

#  ----------------- 2ª Camada Oculta -------------------
#  camada de entrada com 30 neurônios (caracteiristicas como radius_mean, texture_mean, etc.)
classificador.add(Dense(units=16, activation='relu', kernel_initializer='random_uniform'))

#  camada de saída
classificador.add(Dense(units=1, activation='sigmoid'))
#  learning rate - chegar no máximo global
otimizador = keras.optimizers.Adam(learning_rate=0.001, decay=0.0001, clipvalue=0.5)
classificador.compile(optimizer=otimizador, loss='binary_crossentropy', metrics=['binary_accuracy'])
#classificador.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])

#  batch_size = lotes ( de 10 em 10 )
classificador.fit(previsores_treinamento, classe_treinamento, batch_size=10, epochs=100)

#  Pesos que a rede neural encontrou, pesos0[1] = bias
pesos0 = classificador.layers[0].get_weights()
pesos1 = classificador.layers[1].get_weights()
pesos2 = classificador.layers[2].get_weights()
#  --------------------- Medição ----------------------------

#  usando sklearn para medir
previsoes = classificador.predict(previsores_teste)
previsoes = (previsoes > 0.5)

precisao = accuracy_score(classe_teste, previsoes)
matriz = confusion_matrix(classe_teste, previsoes)

#  usando o keras
resultado = classificador.evaluate(previsores_teste, classe_teste)
