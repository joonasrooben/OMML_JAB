import pandas as pd
import scipy
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from NeuralNetwork import *
data = pd.read_csv('DATA.csv')
X_train, X_test, y_train, y_test = train_test_split(data[['x1','x2']], data[['y']], test_size=0.255, random_state=1990243)

X_train = X_train.to_numpy()
X_test = X_test.to_numpy()
y_tr = y_train.to_numpy()
y_te = y_test.to_numpy()

rho = 0.0007
sigma = 1.5
N = 32


nn = NeuralNetwork(2, rho, sigma, N)

v = np.array(nn.createWeightsForHiddenLayer(N))

omega = nn.createOmega()

W = omega[:-2*N].reshape((2, N))
v = omega[-(N):]
bias = omega[-2*N:-N]
init_time = time.time()
result = nn.minimise(nn.MLP, v, args=(X_train, y_tr, W, bias))
optimized_weights = result.x
omega2 = np.append(np.append(W, bias), optimized_weights)
nn2 = NeuralNetwork(1, rho, sigma, N)
omega2 = np.append(np.append(W, bias), optimized_weights)
mlp = nn2.MLP(omega2, X_train, y_tr)
init_time = time.time()
result2 = nn2.minimise(nn2.MLP, omega2, args=(X_train, y_tr))
optimization_time = time.time() - init_time
optimized_weights = result2.x
train_err = nn2.data_error(optimized_weights, X_train, y_tr)
test_err = nn2.data_error(optimized_weights, X_test, y_te)

nr_fun = result2.nfev
nr_gr = result2.njev

s = f"""
{'-'*40}
# N: {32}
# Sigma: {1.5}
# Rho: {0.0007}
# Optimization solver: CG
# Number of function evaluations : {nr_fun}
# Number of gradient evaluations : {nr_gr}
# Time for optimizing the network : {optimization_time}

# Training Error : {train_err}
# Test Error : {test_err}
{'-'*40}"""

print(s)
nn2.plotting(result2.x, optimized_weights, bias, title="Approximating function (MLP)")
