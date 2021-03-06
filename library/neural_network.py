import numpy
import scipy.special
import csv
import datetime
import os
from pathlib import Path

# TODO Softmax Function for Output with cross entropy cost

'''
Neural network that can have n amount of layers and nodes per layer.
It can also have tanh, sigmoid or relu as an activation function for the hidden layers
and for the output layer. Cost function is squared error.

Weights and configuration can be stored and retrieved from a CSV file.
'''


class _ActivationFunction:

    def execute(self):
        pass

    def derivative(self, y):
        pass


class TanhActivationFunction(_ActivationFunction):

    def execute(self, x):
        return numpy.tanh(x)

    def derivative(self, y):
        return (1.0 - (y * y))


class SigmoidActivationFunction(_ActivationFunction):

    def execute(self, x):
        return scipy.special.expit(x)

    def derivative(self, y):
        return (y * (1.0 - y))


class ReluActivationFunction(_ActivationFunction):

    def execute(self, x):
        return x * (x > 0)

    def derivative(self, y):
        return 1. * (y > 0)


class NeuralNetwork:

    def __init__(self, nodes, learningRate=0.1, activation_function=TanhActivationFunction(), output_function=SigmoidActivationFunction(), momentum_factor=0.5):
        self.momentum_factor = momentum_factor
        self.activation_function = activation_function
        self.output_function = output_function
        self.learningRate = learningRate
        self.nodes = nodes
        self.layers = len(nodes)
        self.weights = []
        self.biases = []
        self.biases_momentum = []
        self.momentum = []

        # Create all network layers
        for i in range(1, self.layers):
            # Initialize Layer weights (Output Weights x Input Weights for later matrix dot product for weighted sum).
            self.weights.append(numpy.random.normal(
                0.0, pow(self.nodes[i], -0.5), (self.nodes[i], self.nodes[i-1])))
            # Initialize Momentum and Biases
            self.momentum.append(numpy.zeros(
                (self.nodes[i], self.nodes[i-1])))
            self.biases_momentum.append(
                numpy.zeros((self.nodes[i], 1)))
            self.biases.append(numpy.ones((self.nodes[i], 1)))

    def train(self, inputs, targets):
        ''' Calculates inputs, compares with targets, and trains network '''

        originalInputs = numpy.array(inputs, ndmin=2).T
        inputs = numpy.array(inputs, ndmin=2).T
        targets = numpy.array(targets, ndmin=2).T
        finalOutputs = []

        for i in range(0, self.layers-1):
            if i == self.layers-2:
                # Output Layer function
                inputs = self.output_function.execute(
                    numpy.dot(self.weights[i], inputs) + self.biases[i])
            else:
                # Default activation on hidden layers
                inputs = self.activation_function.execute(
                    numpy.dot(self.weights[i], inputs) + self.biases[i])
            finalOutputs.append(inputs)

        # Calculate errors
        errors = []
        outputErrors = targets - finalOutputs[-1]
        errors.append(outputErrors)

        for i in range(self.layers-2, 0, -1):
            if(i == (self.layers-2)):
                errors.insert(0, numpy.dot(self.weights[i].T, outputErrors))
            else:
                errors.insert(0, numpy.dot(self.weights[i].T, errors[0]))

        # Recalculate weights with back propagation
        # Calculate Gradients --> Derivative of Error function in regards to specific weight
        # Dot product with transposed Output of previous node for weight adjustment
        weightDelta = 0
        gradients = None
        for i in range(self.layers-2, -1, -1):
            # On last node use inputs instead of some outputs
            if i == 0:
                # Calculate gradients from derivative of activation function
                gradients = self.learningRate * \
                    errors[i] * \
                    self.activation_function.derivative(finalOutputs[i])
                weightDelta = numpy.dot(
                    gradients, numpy.transpose(originalInputs))
            elif i == self.layers-2:
                # Output layer, use derivative of output function
                gradients = self.learningRate * \
                    errors[i] * \
                    self.output_function.derivative(finalOutputs[i])
                weightDelta = numpy.dot(gradients,
                                        numpy.transpose(finalOutputs[i-1]))
            else:
                # Calculate gradients from derivative of activation function
                gradients = self.learningRate * \
                    errors[i] * \
                    self.activation_function.derivative(finalOutputs[i])
                weightDelta = numpy.dot(gradients,
                                        numpy.transpose(finalOutputs[i-1]))
            # Add momentum
            self.weights[i] += (weightDelta + self.momentum[i])
            self.biases[i] += (gradients + self.biases_momentum[i])
            # Apply new momentum
            self.momentum[i] = self.momentum_factor * weightDelta
            self.biases_momentum[i] = self.momentum_factor * gradients

    def printSums(self):
        '''
        Prints sum of weights for checking if network is the same after storing/loading
        '''
        print(numpy.sum(numpy.array([plane.sum() for plane in self.weights])))

    def query(self, inputs):
        ''' Get output for input '''

        inputs = numpy.array(inputs, ndmin=2).T

        for i in range(0, self.layers-1):
            if i == self.layers-2:
                # Output Layer function
                inputs = self.output_function.execute(
                    numpy.dot(self.weights[i], inputs) + self.biases[i])
            else:
                # Default activation on hidden layers
                inputs = self.activation_function.execute(
                    numpy.dot(self.weights[i], inputs) + self.biases[i])
        return inputs

    def storeResult(self):
        ''' Store weights, configuration in csv file '''
        script_dir = os.path.dirname(__file__)
        filenameWeights = "./nn_weights"
        filenameBiases = "./nn_biases"
        pathWeights = os.path.join(script_dir, filenameWeights)
        pathBiases = os.path.join(script_dir, filenameBiases)

        # Store weights
        numpy.save(pathWeights, self.weights)
        # Store biases
        numpy.save(pathBiases, self.biases)

    def loadResult(self):
        ''' Load previous results and configure network '''
        script_dir = os.path.dirname(__file__)
        filenameWeights = "./nn_weights.npy"
        filenameBiases = "./nn_biases.npy"
        pathWeights = os.path.join(script_dir, filenameWeights)
        pathBiases = os.path.join(script_dir, filenameBiases)

        # Load weights
        self.weights = numpy.load(pathWeights)
        # Load biases
        self.biases = numpy.load(pathBiases)
