import pandas as pd
import scipy
import numpy as np
import math
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class NeuralNetwork(object):
    """
    A Optimizer class.

    """
    def __init__(self, rho, sigma, N):
        """
        Inputs:
        - task: solvable task
        - rho: rho
        - sigma: sigma
        - N: nr of neurons
        """
        self.rho = rho
        self.sigma = sigma
        self.N = N

    def activationFunction(self, t, sigma = 1):
        """
        This function is the activation function tanh with two parameters. To optimise our model we have to vary the sigma.
        :param t: input
        :param sigma: the spread σ
        :return:
        """
        return (np.exp(2*self.sigma*t)-1)/(np.exp(2*self.sigma*t)+1)

    def createWeightsForOneNeuron(self, N):
        """
        We want to create a vector of weight using the “xavier” initialization.
        The current standard approach for initialization of the weights of neural network layers and nodes that use the Sigmoid or TanH activation function is called “glorot” or “xavier” initialization

        The xavier initialization method is calculated as a random number with a uniform probability distribution (U) between the range -(1/sqrt(n)) and 1/sqrt(n), where n is the number of inputs to the node.

        :param N: the number of neurons in the hidden layer
        :return: a array of weights
        """
        initial_weights_one_neuron = []
        for i in range(0, N):
            initial_weights_one_neuron.append(np.random.uniform(low = -(1/math.sqrt(N)), high = 1/math.sqrt(N)))
        return initial_weights_one_neuron

    def create_initial_weights(self, N):
        """
        We want to crate the 2D-matrix of weights "W".
        :param N: the number of neurons in the hidden layer
        :return: a 2D-array of weights
        """
        initial_weights = []
        for i in range(0, 2):
            initial_weights.append(self.createWeightsForOneNeuron(N))
        return initial_weights

    def multInputWeights(self, input, initialWeights):
        """
        We make here the multiplication of the inputs (X) with the weights (W) --> X*W  where X is (n,2) and W is (2,N).
        So we end up with a matrix mult (n,N).
        :param input: inputs
        :param initialWeights: weights of the input
        :return: 2D-array (n,N) dimension
        """
        return np.matmul(input, initialWeights)

    def createBias(self, N):
        """
        Creation vector of bias
        :param N: the number of neurons in the hidden layer
        :return: vector of bias
        """
        return np.ones((N, 1))

    def computeLinearResult(self, mult, bias):
        """
        return the linear regression
        :param mult: multiplication of inputs and weights
        :param bias: vector of bias
        :return:
        """
        return mult + bias

    def createWeightsForHiddenLayer(self, N):
        """
        We want to create a vector of weight using the “xavier” initialization. the vector of weight is V.
        :param N: the number of neurons in the hidden layer
        :return: a array of weights
        """
        weightHiddenLayer = []
        for i in range(0, N):
            weightHiddenLayer.append(np.random.uniform(low = -(1/math.sqrt(N)), high = 1/math.sqrt(N)))
        return weightHiddenLayer

    def computeLinearOutput(self, v, activationOutput):
        """
         Compute the output of the neural network.

        :param v: weights of the hidden layer
        :param activationOutput: output of the activation function
        :return:
        """
        return np.matmul(v, activationOutput.T)

    def computeEmpiricalRisk(self, target, output, P):
        """
        Compute empirical risk.

        :param target: this will be the y or f(x1,x2)
        :param output: output of the model
        :param P: number of observations
        :return:
        """
        return (1/(2*P))*sum((output - target)**2)

    def computeRegularizationTerm(self, omega, rho = 1):
            """
            Compute the regularization term for the vector ω = (v,w,b).
            :param omega: ω = (v,w,b).
            :return:
            """
            normV = np.linalg.norm(omega[0]**2)
            normW = np.linalg.norm(omega[1]**2)
            normB = np.linalg.norm(omega[2]**2)

            omega_norm = normV + normW + normB

            return (rho/2)*(omega_norm)

    def createOmega(self):
        W = np.array(self.create_initial_weights(self.N))
        v = np.array(self.createWeightsForHiddenLayer(self.N))
        bias = self.createBias(self.N)
        omega = np.append(np.append(W.flatten(), bias), v)

        return omega

    def MLP(self, omega ,input, target):

        W = omega[:-2*(self.N)].reshape((2, (self.N)))
        v = omega[-(self.N):]
        bias = omega[-2*(self.N):-(self.N)]

        # X*W
        mult = self.multInputWeights(input, W)
        # sum(X*W) + b
        linearResult = self.computeLinearResult(mult, bias)
        # g(sum(X*W) + b)
        acti = self.activationFunction(linearResult, self.sigma)
        # output of the model : sum(v* g(sum(X*W) + b))
        f_x = self.computeLinearOutput(v, acti)
        # reshape the target
        y = target
        y = target.reshape((-1))
        # number of observations
        P = len(input)
        expectationRisk = self.computeEmpiricalRisk(y, f_x, P)
        omega = [v, W, bias]
        regularizationTerm = self.computeRegularizationTerm(omega, self.rho)

        return expectationRisk + regularizationTerm


    def minimise(self, method, omega, args, maxiters = 300):
        """

        :param method: method that we use to do the neural network
        :param omega: vector omega with the weights and the bias
        :param args: arguments of the function "method"
        :param maxiters: max iterations for the minimize method.
        :return:
        """
        return scipy.optimize.minimize(method, omega, args, method = 'CG', options={"maxiter":maxiters})

    def prediction(self, optimizedWeights, meshgrid1D):
        W = optimizedWeights[:-2* self.N].reshape((2,self.N))
        v = optimizedWeights[-self.N:]
        bias = optimizedWeights[-2*self.N:-self.N]

        mult = self.multInputWeights(meshgrid1D, W)
        linearResult = self.computeLinearResult(mult, bias)
        acti = self.activationFunction(linearResult, self.sigma)
        f_x = self.computeLinearOutput(v, acti)

        return f_x

    def plotting(self, weightOptimized, title='Plotting of the function'): #if you do not provide a title, 'Plotting...' will be used
            #create the object
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        n = 100

        #create the grid
        x = np.linspace(-2, 2, n) #create 50 points between [-5,5] evenly spaced
        y = np.linspace(-3, 3, n)
        X, Y = np.meshgrid(x, y)
        mixed =  np.vstack((X.flatten(),Y.flatten())).T
        #make the nxn grid into 2d array that can be fed into func
        Z = self.prediction(weightOptimized, mixed)#evaluate the function (note that X,Y,Z are matrix)
        Z = Z.reshape((n,n))

        ax.plot_surface(X, Y, Z, rstride=1, cstride=1,cmap='viridis', edgecolor='none')

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title(title)
        plt.show()
