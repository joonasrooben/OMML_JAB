
import NeuralNetwork as nn
import numpy as np


def ICanGeneralize(X, optimizedWeights) :
    X = np.array(X)
    X1 = X[:,0]
    X2 = X[:,1]
    mixed =  np.vstack((X1.flatten(),X2.flatten())).T
    Y_pred = nn.prediction(optimizedWeights, mixed)

    return Y_pred