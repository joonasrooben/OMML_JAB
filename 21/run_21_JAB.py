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

rho = 1e-5
sigma = 1
N = 20

print("1")
nn = NeuralNetwork(2, rho, sigma, N)
print("2")
v = np.array(nn.createWeightsForHiddenLayer(N))
print("3")
omega = nn.createOmega()
print("4")
W = omega[:-2*N].reshape((2, N))
print("5")
bias = omega[-2*N:-N]
print("6")
print(omega)
print("7")
mlp = nn.MLP(omega, X_train, y_tr, W, bias)
print("8")
init_time = time.time()
print("9")
result = nn.minimise(nn.MLP, v, args=(X_train, y_tr, W, bias))
print("10")
optimized_weights = result.x
print("11")
print("###################")
print("W shape : ", W.shape)
print("bias shape : ", bias.shape)
print("v shape : ", optimized_weights.shape)
omega2 = np.append(np.append(W.flatten(), bias), optimized_weights)

print(omega2.shape)
result2 = nn.minimise(nn.MLP, omega2, args=(X_train, y_tr))
print("------------------")
optimization_time = time.time() - init_time
optimized_weights = result2.x
train_err = nn.data_error(optimized_weights, X_train, y_tr)
test_err = nn.data_error(optimized_weights, X_test, y_te)

nr_fun = result.nfev
nr_gr = result.njev

s = f"""
{'-'*40}
# N: {32}
# Sigma: {1.0}
# Rho: {0.0009}
# Optimization solver: CG
# Number of function evaluations : {nr_fun}
# Number of gradient evaluations : {nr_gr}
# Time for optimizing the network : {optimization_time}

# Training Error : {train_err}
# Test Error : {test_err}
{'-'*40}"""

print(s)
nn.plotting(result2.x,  title="function with MLP using Extreme learning")
