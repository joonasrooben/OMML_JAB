
from run_11_JAB import *

#we use weights calculated in the run file
W = optimized_weights
N=32
sigma = 1.5
rho = 0.0007

def ICanGeneralize(X) :
    X = np.array(X)
    X1 = X[:,0]
    X2 = X[:,1]
    mixed =  np.vstack((X1.flatten(),X2.flatten())).T
    Y_pred = nn.prediction(W, mixed)
    return Y_pred