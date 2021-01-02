import pandas as pd
from keras.models import Sequential  # Modelo Sequencial: uma camada após a outra
from keras.layers import Dense, Dropout  # Fully conected: um neuronio conectado a todos os outros
from sklearn.preprocessing import LabelEncoder  # String to number
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

base = pd.read_csv('Multiclass Classification/Iris Dataset/iris.csv')
previsores = base.iloc[:, 0:4].values
classe = base.iloc[:, 4].values

labelencoder = LabelEncoder()
classe = labelencoder.fit_transform(classe)
classe_dummy = np_utils.to_categorical(classe)

def criar_rede():
    classificador = Sequential()
    # Quantidade de neuronios na camada oculta = (quant. entradas(4) + quant. saidas(3)) / 2
    # input_dim = quant. de previsores
    # 2 camadas ocultas:
    classificador.add(Dense(units=4, activation='relu', input_dim=4))
    classificador.add(Dense(units=4, activation='relu'))
    # Camada de saída
    classificador.add(Dense(units=3, activation='softmax'))

    classificador.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    return classificador

classificador = KerasClassifier(build_fn = criar_rede, epochs=1000, batch_size=10)

#  cv=10 === 10-fold cross. Gerando assim 10 resultados
resultados = cross_val_score(estimator=classificador, X=previsores, y=classe, cv=10, scoring='accuracy')

media = resultados.mean()
#  Desvio Padrão - verificar se tem muito overfitting
desvio = resultados.std()
