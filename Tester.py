import pandas as pd
import scipy
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from Optimizer import *
data = pd.read_csv('DATA (1).csv')
X_train, X_test, y_train, y_test = train_test_split(data[['x1','x2']], data[['y']], test_size=0.255, random_state=1990243)

X_train = X_train.to_numpy()
X_test = X_test.to_numpy()
y_tr = y_train.to_numpy()
y_te = y_test.to_numpy()

## gradients are always taken into account to speed up the calculations (in every task)
## an example how to use it: 
opt = Optimizer(3, 1e-3, 0.5, 50, 'CG')
result = opt.minimize(X_train, y_tr, 112)

print(result[0])
test_err = opt.test_loss(result[1],X_test, y_te)
print(test_err)
opt.plotting(result[1], 'The approximator')
##### end of an example

## an example how to use it for task 3: 
options = {"maxiter" : 20, "gtol" : 1e-03} # use for the third task
opt = Optimizer(3, 1e-5, 0.5, 64, 'L-BFGS-B')
result = opt.minimize(X_train, y_tr, 112, opts = options, loops = 1e-03) #use for the third tas
print(result[0].message)
train_err = opt.test_loss(result[1], X_train,y_tr)
test_err = opt.test_loss(result[1],X_test, y_te)
print("Final test error:", test_err,"FInal train error:", train_err)
opt.plotting(result[1], 'The approximator')
##### end of an example



