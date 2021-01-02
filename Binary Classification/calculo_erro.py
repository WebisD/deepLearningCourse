import numpy as np

def meanSquareError(classe, calculado):
    n = len(classe)
    y = classe - calculado
    y = y**2
    mse = (1/n) * (y.sum())

    return mse

def rootMeanSquareError(classe, calculado):
    mse = meanSquareError(classe, calculado)
    rmse = np.sqrt(mse)

    return rmse

def meanAbsoluteError(classe, calculado):
    y = np.abs(classe - calculado)
    n = len(classe)

    mae = (1/n) * (y.sum())

    return mae

def taxaDeAcerto(classe, calculado):
    mae = meanAbsoluteError(classe, calculado)
    tErro = mae * 100
    tAcerto = 100 - tErro

    return tAcerto


classe = np.array([1, 0, 1, 0])
calculado = np.array([0.300, 0.020, 0.890, 0.320])

print("MSE:", meanSquareError(classe, calculado))
print("RMSE:", rootMeanSquareError(classe, calculado))
print("MAE:", meanAbsoluteError(classe, calculado))
print("Taxa de Acerto:", taxaDeAcerto(classe, calculado))