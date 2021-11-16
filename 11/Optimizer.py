import pandas as pd
import scipy
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Optimizer(object):
    """
    A Optimizer class. 

    """
    def __init__(self, task, rho, sigma, N, met):
        """
        Inputs:
        - task: solvable task
        - rho: rho
        - sigma: sigma
        - N: nr of neurons
        - met: method
        """
        self.task = task
        self.rho = rho
        self.sigma = sigma
        self.N = N
        self.met = met

    def minimize(self, data, labels, seed=123, tol = None, opts = None, loops = 0.001):
        """
        Minimizer task.

        Inputs:
        - data: train or test data
        - labels: labels
        - seed: by defult 123
        - tol: the tolerance of optimizer (default None)
        - opts: special options passed as a dict to optimizer e.g {'maxiter':100, 'gtol':1e-05} (default None)
        - loops: the difference to be achieved between two itertions in decomposition method in task 3 (default 0.001)
        
        Returns: 
        - model with info
        - learned weights in default setting
        - initial loss
        - the model also outputs the stats in list = [initial_loss, iterations, nr_grads_ev, nr_fun_ev]
            (NB in the task #3 the iterations is the nr of outer iterations)
        """
        

        if self.task == 1:
            C,V,res = self.preproc(self.N, data, seed = seed)
            shape = C.shape
            args = (self.N,self.sigma, res, labels, self.rho, shape, None, 0) #three modes: 0: both supervised; 1: centers unsupervised; 2:weights v unsupervised
            W = np.append(C.flatten(),V)
            initial_loss = self.loss(W, args)
            result_rbf = scipy.optimize.minimize(self.loss, W,jac = self.grad_both, args= (args,),
                                                 method=self.met, options = opts, tol = tol)
            W = result_rbf.x
            nr_grads = result_rbf.njev
            nr_fun = result_rbf.nfev
            iters = result_rbf.nit
            
        elif self.task == 2:
            C,V,res = self.preproc(self.N, data, 1, seed=seed)
            shape = C.shape
            args = (self.N,self.sigma, res, labels, self.rho, shape,C, 1) #three modes: 0: both supervised; 1: centers unsupervised; 2:weights v unsupervised
            W = V 
            initial_loss = self.loss(W, args)
            result_rbf = scipy.optimize.minimize(self.loss, W, args= (args,),jac=self.grad_v,
                                                  method=self.met, options = opts, tol = tol)
            W = np.append(C, result_rbf.x)
            nr_grads = result_rbf.njev
            nr_fun = result_rbf.nfev
            iters = result_rbf.nit
            
        else:
            C,V,res = self.preproc(self.N, data, 1, seed = seed)
            shape = C.shape
            args = (self.N,self.sigma, res, labels, self.rho, shape,C, 1)
            W = V 
            initial_loss =  self.loss(W, args)
            print('Loss in the beginning:', initial_loss)            
            a = initial_loss
            diff = 10e10
            nr_grads = 0
            nr_fun = 0
            iters = 0
            while (diff > loops):
                iters += 1
                result_rbf = scipy.optimize.minimize(self.loss, W, args= (args,) ,jac=self.grad_v,
                                                     method=self.met)#, options = opts, tol = tol)
                nr_grads += result_rbf.njev
                nr_fun += result_rbf.nfev
                print('Loss after convex optim:',result_rbf.fun)
                V = result_rbf.x
                args = (self.N,self.sigma, res, labels, self.rho, shape,result_rbf.x, 2)
                W = C
                result_rbf = scipy.optimize.minimize(self.loss, W, args= (args,) ,jac = self.grad_c, 
                                                     method=self.met, options = opts, tol = tol)
                nr_grads += result_rbf.njev
                nr_fun += result_rbf.nfev
                b = result_rbf.fun
                print('Loss after non convex optim:',b)
                args = (self.N, self.sigma, res, labels, self.rho, shape,result_rbf.x, 1)
                W =  V
                diff = a - b
                a = b

            W = np.append(result_rbf.x, W)
        return result_rbf, W, [initial_loss, iters, nr_grads, nr_fun]


    def test_loss(self,W, data, labels):
        """
        Testing function

        Inputs:
        - W: weghts on default shape
        - data: test data
        - labels: labels of test data
        """

        a,b, data_x = self.preproc(self.N, data)
        data_y = labels 

        args = (self.N, self.sigma, data_x, data_y, self.rho, None)
        ter = self.test_error(W, args)
        return ter
 

    def plotting(self,W, title='Plotting of the function'): #if you do not provide a title, 'Plotting...' will be used
        #create the object
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        n = 100
        #create the grid
        x = np.linspace(-2, 2, n) #create 50 points between [-5,5] evenly spaced  
        y = np.linspace(-3, 3, n)
        X, Y = np.meshgrid(x, y) #create the grid for the plot
        mixed =  np.vstack((X.flatten(),Y.flatten())).T
        a,b,res = self.preproc(self.N,mixed)
        Z = self.func(W, (self.N, self.sigma,res,None, self.rho ))#evaluate the function (note that X,Y,Z are matrix)
        Z = Z.reshape((n,n))

        ax.plot_surface(X, Y, Z, rstride=1, cstride=1,cmap='viridis', edgecolor='none')

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title(title)
        plt.show()


    def preproc(self, N, X_train, hold = False, seed = 123): #n: the nr of neurons; hold: if we want the centers to be chosen from X_train
        np.random.seed(seed)
        if hold == True:
            a = X_train[np.random.choice(np.arange(X_train.shape[0]), N, replace=False)].reshape(1,N,2) 
        else:
            a =  np.random.normal(size=(1, N, 2))

        res = np.repeat(X_train,N,axis=0).reshape(X_train.shape[0],N,2)
        
        v = np.random.normal(size = N)
        return a,v,res # a:centers;v: weights; res: train data with compatible shape

    def Gaussian(self, X, sigma):
        return np.exp(-(np.linalg.norm(X,axis = -1)/sigma)**2)
        
    def grad_c(self, W, args):
        N = args[0]
        v = args[-2]
        a = W.reshape(1,N,2) 


        y = args[3]
        c = np.tile(a,(args[2].shape[0],1)).reshape(args[2].shape[0],N,2)
        main = self.RBF((c,v), args) - y.T  
        dervs = np.ones((N, 2)) 
        for i, vk in enumerate(v):
            buff = np.transpose(args[2], (1,0,2))[0] - np.repeat(np.array([c[0][i]]), len(args[2]), axis=0)
            buff2 = main @ (np.repeat(self.Gaussian(buff, args[1]),2,axis =0).reshape((args[2].shape[0],2)) * 2 * (buff)/(np.linalg.norm(buff)*args[1]**2))
            
            dervs[i] = vk * buff2 + args[4]* c[0][i]
        return dervs.flatten()

    def grad_both(self, W,args):
        N = args[0]
        v = W[-N:]
        a = W[:-N]
        args1 = (args[0],args[1], args[2], args[3], args[4], args[5],a.reshape(1,N,2), 0)
        v_g = self.grad_v(v,args1)
        args2 = (args[0],args[1], args[2], args[3], args[4], args[5],v, 0)
        c_g = self.grad_c(a,args2)
        return np.append(c_g.flatten(),v_g)
  
    def grad_v(self, W, args):
        N = args[0]
        a = args[-2]
        v = W
        y = args[3]
    #    c = np.repeat(a,args[2].shape[0],axis=1).reshape((args[2].shape[0],N,2)).T
        c = np.tile(a,(args[2].shape[0],1)).reshape(args[2].shape[0],N,2)
        phi = self.Gaussian(args[2]-c, args[1])
        grads = phi.T @ (phi @ v - y.reshape(-1)) + len(args[2])*args[4]*v
        return grads

    def loss(self, W, args):
        N = args[0]
        if args[-1] == 1:
            a = args[-2]
            v = W
            regu = np.sum(v**2)
        elif args[-1] == 2:
            v = args[-2]
            a = W.reshape(1,N,2) 
            regu = np.sum(a**2) 
        else:
            v = W[-N:]
            a = W[:-N].reshape(1,N,2)
            regu = np.sum(a**2) + np.sum(v**2)
        y = args[3]
        c = np.tile(a,(args[2].shape[0],1)).reshape(args[2].shape[0],N,2)

        #regu = np.sum(a**2) + np.sum(v**2)
        loss = float((1/(2*len(args[2]))* np.sum((self.RBF((c,v), args) - y.T)**2, axis=1) + args[4]/2* regu)[0])
        return loss 

    def RBF(self, W, args):
        sigma = args[1]
        X = args[2]
        C = W[0]
        v = W[1]
        #print(v.shape, Gaussian(X-C,sigma).shape)
        return np.matmul(v,self.Gaussian(X-C, sigma).T) 
# sigma is a RBF spread parameter and simga > 0

    def func(self, W, args): #for plotting
        N = args[0]
        v = W[-N:]
        a = W[:-N].reshape(1,N,2)
        #c = np.repeat(a,args[2].shape[0],axis=1).reshape((args[2].shape[0],N,2))
        c = np.tile(a,(args[2].shape[0],1)).reshape(args[2].shape[0],N,2)


        return self.RBF((c,v), args)

    def test_error(self, W,  args):
        N = args[0]
        v = W[-N:]
        a = W[:-N].reshape(1,N,2)
        y = args[3]
        #c = np.repeat(a,args[2].shape[0],axis=1).reshape((args[2].shape[0],N,2))
        c = np.tile(a,(args[2].shape[0],1)).reshape(args[2].shape[0],N,2)


        return float((1/(2*len(args[2]))* np.sum((self.RBF((c,v), args) - y.T)**2, axis=1)))

    
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
    