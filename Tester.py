import pandas as pd
import scipy
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

from NeuralNetwork import *
from Optimizer import *
data = pd.read_csv('DATA (1).csv')
X_train, X_test, y_train, y_test = train_test_split(data[['x1','x2']], data[['y']], test_size=0.255, random_state=1990243)

X_train = X_train.to_numpy()
X_test = X_test.to_numpy()
y_tr = y_train.to_numpy()
y_te = y_test.to_numpy()

## gradients are always taken into account to speed up the calculations (in every task)
## an example how to use it: 
#opt = Optimizer(3, 1e-3, 0.5, 50, 'CG')
#result = opt.minimize(X_train, y_tr, 112)
#print(result[0])
#test_err = opt.test_loss(result[1],X_test, y_te)
#print(test_err)
#opt.plotting(result[1], 'The approximator')
##### end of an example
"""
Exercice 1
"""
rho = 1e-5
sigma = 1
N = 20

#Exercice 1

nn = NeuralNetwork(1, rho, sigma, N)
omega = nn.createOmega()
mlp = nn.MLP(omega, X_train, y_tr)
result = nn.minimise(nn.MLP, omega, args=(X_train, y_tr))
nn.plotting(result.x)


#Exercice 2

nn = NeuralNetwork(2, rho, sigma, N)
v = np.array(nn.createWeightsForHiddenLayer(N))
omega = nn.createOmega()
W = omega[:-2*N].reshape((2,N))
bias = omega[-2*N:-N]
mlp = nn.MLP(omega, X_train, y_tr, W, bias)
result = nn.minimise(nn.MLP, v, args=(X_train, y_tr, W, bias))
nn.plotting(result.x, W, bias)


"""
def grid_search(X_train, X_test, y_tr, y_te, task, rho_list, sigma_list, N_list, met) :
    opt_sigma = -1
    opt_rho = -1
    opt_N = -1
    opt_test_err = 1000

    hyperparams_recap = np.zeros(shape=(len(rho_list),
                                        len(sigma_list),
                                        len(N_list)))
    for rho in range(len(rho_list)) :
        for sigma in range(len(sigma_list)) :
            for N in range(len(N_list)) :
                opt = Optimizer(task,
                                rho_list[rho],
                                sigma_list[sigma],
                                N_list[N],
                                met)
                result = opt.minimize(X_train, y_tr, 112)
                test_err = opt.test_loss(result[1], X_test, y_te)

                hyperparams_recap[rho, sigma, N] = test_err

                if(test_err < opt_test_err) :
                    opt_sigma = sigma_list[sigma]
                    opt_rho = rho_list[rho]
                    opt_N = N_list[N]
                    opt_test_err = test_err
    return ((opt_rho, opt_sigma, opt_N),hyperparams_recap)
"""

