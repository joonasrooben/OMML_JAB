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
nn = NeuralNetwork(1, rho, sigma, N)
omega = nn.createOmega()
mlp = nn.MLP(omega, X_train, y_tr)
init_time = time.time()
result = nn.minimise(nn.MLP, omega, args=(X_train, y_tr))
optimization_time = time.time() - init_time
optimized_weights = result.x
train_err = nn.data_error(optimized_weights, X_train, y_tr)
test_err = nn.data_error(optimized_weights, X_test, y_te)


nr_fun = result.nfev
nr_gr = result.njev


plotMLP = nn.plotting(result.x, title= "plot of the function using MLP")

s = f"""
{'-'*40}
# N: {30}
# Sigma: {1.0}
# Rho: {1e-5}
# Optimization solver: CG
# Number of function evaluations : {nr_fun}
# Number of gradient evaluations : {nr_gr}
# Time for optimizing the network : {optimization_time}
# Training Error : {train_err}
# Test Error : {test_err}
# The plot is above
{'-'*40}
"""

print(s)