import numpy as np

def stepFunction(soma):
    if soma >= 1:
        return 1
    return 0

def sigmoidFunction(soma):
    return 1/(1+np.exp(-soma))

def tanFunction(soma):
    return (np.exp(soma) - np.exp(-soma)) / (np.exp(soma) + np.exp(-soma))

def ReLUFunction(soma):
    if soma >= 0:
        return soma
    return 0

def linearFunction(soma):
    return soma

def softmaxFunction(x):
    ex = np.exp(x)
    return ex / ex.sum()


print("Step Function:", stepFunction(30))
print("Sigmoid:", sigmoidFunction(2.1))
print("Tangente Hiperbólica:", tanFunction(2.1))
print("ReLU:", ReLUFunction(2.1))
print("Linear:", linearFunction(2.1))

valores = [5.0, 2.0, 1.3]
print("Softmax:", softmaxFunction(valores))
