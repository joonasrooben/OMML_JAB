from run_22_JAB import *

#we use weights calculated in the run file
W = result[1]
N=32
sigma = 1.0
rho = 0.0009

def ICanGeneralize(X) :
    X = np.array(X)
    X1 = X[:,0]
    X2 = X[:,1]
    mixed =  np.vstack((X1.flatten(),X2.flatten())).T
    a,b,res = opt.preproc(N,mixed)
    Y_pred = opt.func(W, (N, sigma,res,None, rho ))
    
    return Y_pred