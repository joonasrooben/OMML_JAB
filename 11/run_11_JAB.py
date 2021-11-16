import pandas as pd
import scipy
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from Optimizer import *
data = pd.read_csv('DATA.csv')
X_train, X_test, y_train, y_test = train_test_split(data[['x1','x2']], data[['y']], test_size=0.255, random_state=1990243)

X_train = X_train.to_numpy()
X_test = X_test.to_numpy()
y_tr = y_train.to_numpy()
y_te = y_test.to_numpy()


opt = Optimizer(1, 0.0009, 1.0, 32, 'CG')

init_time = time.time()
result = opt.minimize(X_train, y_tr, 123)
optimization_time = time.time() - init_time

nr_fun = result[2][3]
nr_gr = result[2][2]
train_err = opt.test_loss(result[1], X_train,y_tr)
test_err = opt.test_loss(result[1],X_test, y_te)



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

{'-'*40}
"""

print(s)