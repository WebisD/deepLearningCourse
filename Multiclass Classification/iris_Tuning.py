import pandas as pd
from keras.models import Sequential  # Modelo Sequencial: uma camada após a outra
from keras.layers import Dense, Dropout  # Fully conected: um neuronio conectado a todos os outros
from sklearn.preprocessing import LabelEncoder  # String to number
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV 

base = pd.read_csv('Multiclass Classification/Iris Dataset/iris.csv')
previsores = base.iloc[:, 0:4].values
classe = base.iloc[:, 4].values

labelencoder = LabelEncoder()
classe = labelencoder.fit_transform(classe)


def criar_rede(optimizer, activation, kernel_initializer, neurons, dropout):
    classificador = Sequential()
    # Quantidade de neuronios na camada oculta = (quant. entradas(4) + quant. saidas(3)) / 2
    # input_dim = quant. de previsores
    # 2 camadas ocultas:
    classificador.add(Dense(units=neurons, activation=activation, kernel_initializer=kernel_initializer, input_dim=4))
    classificador.add(Dropout(dropout))

    classificador.add(Dense(units=neurons, activation=activation, kernel_initializer=kernel_initializer))
    classificador.add(Dropout(dropout))
    # Camada de saída
    classificador.add(Dense(units=3, activation='softmax'))

    classificador.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return classificador


classificador = KerasClassifier(build_fn=criar_rede)
parametros={
    'batch_size': [10, 20, 30],
    'epochs': [1000, 1500, 2000],
    'optimizer': ['adam', 'sgd'],
    'kernel_initializer': ['random_uniform', 'normal'],
    'activation': ['relu', 'tanh'],
    'neurons': [2, 4, 8],
    'dropout': [0.2, 0.3],
}

grid_search = GridSearchCV(estimator=classificador, param_grid=parametros, cv=10)
grid_search = grid_search.fit(previsores, classe)

melhores_parametros = grid_search.best_params_
melhor_precisao = grid_search.best_score_

