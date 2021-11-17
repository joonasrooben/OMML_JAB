from run_11_JAB import *

W = result[1]
N=32
sigma = 1.0
rho = 0.0009

def ICanGeneralize(X) :
    return X
    X1 = X[:,0]
    X2 = X[:,1]
    mixed =  np.vstack((X1.flatten(),X2.flatten())).T
    a,b,res = opt.preproc(N,mixed)
    Y_pred = opt.func(W, (N, sigma,res,None, rho ))

    return Y_pred 