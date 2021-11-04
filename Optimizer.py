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

    def minimize(self, data, labels, seed=123):
        """
        Minimizer task.

        Inputs:
        - data: train or test data
        - labels: labels
        - seed: by defult 123

        Returns: 
        - model with info
        - learned weights in default setting
        """
        

        if self.task == 1:
            C,V,res = self.preproc(self.N, data, seed = seed)
            shape = C.shape
            args = (self.N,self.sigma, res, labels, self.rho, shape, None, 0) #three modes: 0: both supervised; 1: centers unsupervised; 2:weights v unsupervised
            W = np.append(C.flatten(),V)
            print('Loss in the beginnnig:', self.loss(W, args))
            result_rbf = scipy.optimize.minimize(self.loss, W,jac = self.grad_both, args= (args,) , method=self.met)#, options = {"maxiter": 500})
            W = result_rbf.x
        elif self.task == 2:
            C,V,res = self.preproc(self.N, data, 1, seed=seed)
            shape = C.shape
            args = (self.N,self.sigma, res, labels, self.rho, shape,C, 1) #three modes: 0: both supervised; 1: centers unsupervised; 2:weights v unsupervised
            W = V 
            print('Loss in the beginning:', self.loss(W, args))
            result_rbf = scipy.optimize.minimize(self.loss, W, args= (args,),jac=self.grad_v,  method=self.met)#, options = {"maxiter": 5})
            W = np.append(C, result_rbf.x)
        else:
            C,V,res = self.preproc(self.N, data, 1, seed = seed)
            shape = C.shape
            args = (self.N,self.sigma, res, labels, self.rho, shape,C, 1)
            W = V 
            print('Los in the beginng:', self.loss(W, args))
            result_rbf = scipy.optimize.minimize(self.loss, W, args= (args,) ,jac=self.grad_v, method=self.met)
            print(result_rbf)
            V = result_rbf.x
            args = (self.N,self.sigma, res, labels, self.rho, shape,result_rbf.x, 2)
            W = C
            print('Loss after non convex optim:', self.loss(W, args))
            result_rbf = scipy.optimize.minimize(self.loss, W, args= (args,) ,jac = self.grad_c, method=self.met)
            W = np.append(result_rbf.x, V)
        return result_rbf, W

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
            buff2 = main @ (np.repeat(self.Gaussian(buff, args[1]),2,axis =0).reshape((args[2].shape[0],2)) * 2 * (buff)/(args[1]**2))
            
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

    