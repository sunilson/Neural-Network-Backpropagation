import sys
sys.path.insert(0, '../library/')
from neural_network import NeuralNetwork, TanhActivationFunction, SigmoidActivationFunction, ReluActivationFunction
import numpy
from matplotlib import pyplot as plt
import scipy.io as scipy
import csv
import matplotlib.pyplot
import datetime
import random
import matplotlib.pyplot as plt


'''
This file is used for testing the network.

Train and test data can be found here: http://yann.lecun.com/exdb/mnist/ https://www.nist.gov/itl/iad/image-group/emnist-dataset

Data is converted to numpy arrays and scaled/normalized before used in training.

Accuracy with MNIST digits is around 96-97%
EMNIST 47 balanced and merged character set is around 80%

There is also a small server that takes black on white images and runs it through
the network and an Android app that lets you draw and send such an image to the server.
Does only work well with the digit set, so the server only has trained data for digits
'''

allresults = []
wrong_guesses = numpy.zeros(10)
img = None


def showImage(pixelArray, delay):
    global img
    if img is None:
        img = matplotlib.pyplot.imshow(pixelArray.reshape(
            (28, 28)), interpolation="None")
    else:
        # Reset flattend array back to 28x28
        img.set_data(pixelArray.reshape((28, 28)))
        matplotlib.pyplot.pause(delay)
        matplotlib.pyplot.draw()


def scaleData(dataArray):
    ''' Scale data respective to max and min values '''
    min = numpy.amin(dataArray)
    max = numpy.amax(dataArray)
    for i in range(len(dataArray)):
        dataArray[i] = (dataArray[i] - min) / (max - min)
    return dataArray

# TRAINING WITH EMINST DATASET


def executeEminstTraining(nn, epoches, runs=-1, offset=-1):

    # Shuffle CSV (doesnt work with full dataset)
    lines = open("./data/emnist/emnist-digits-train.csv", 'r').readlines()
    random.shuffle(lines)
    open("./data/emnist/emnist-digits-train.csv", 'w').writelines(lines)

    now = datetime.datetime.now()
    for i in range(epoches):
        # Read row by row because file is too big
        with open("./data/emnist/emnist-digits-train.csv", 'r') as csvfile:
            reader = csv.reader(csvfile)
            for j, row in enumerate(reader):
                # Apply offset
                if j < offset:
                    continue
                # Stop if max runs is reached
                if runs != -1 and j - offset > runs:
                    break
                print("Training data set " +
                      str(j) + " in epoche " + str(i+1))
                # Rotate image with transpose (original images are rotated)
                rawData = numpy.asfarray(row[1:]).reshape(
                    (28, 28)).transpose().flatten()
                inputs = scaleData(rawData)
                # Sigmoid output is from 0 to 1
                targets = numpy.zeros(10)
                #showImage(rawData, 1)
                # Train
                targets[int(row[0])] = 1.0
                nn.train(inputs, targets)


def executeEminstTest(nn, runs=-1):

    # Shuffle CSV (doesnt work with full dataset)
    lines = open("./data/emnist/emnist-digits-test.csv", 'r').readlines()
    random.shuffle(lines)
    open("./data/emnist/emnist-digits-test.csv", 'w').writelines(lines)

    global allresults
    with open("./data/emnist/emnist-digits-test.csv", 'r') as csvfile:
        reader = csv.reader(csvfile)
        wrong_guess = 0
        counter = 0
        for i, row in enumerate(reader):
            if runs != -1 and i > runs:
                break
            correct_label = int(row[0])
            rawData = numpy.asfarray(row[1:]).reshape(
                (28, 28)).transpose().flatten()
            inputs = scaleData(rawData)
            outputs = nn.query(inputs)
            label = numpy.argmax(outputs)
            if (label != correct_label):
                wrong_guess += 1
                wrong_guesses[correct_label] += 1
                print("Wrong! Guessed: " + str(label) +
                      " inestead of " + str(correct_label))
            else:
                print("Correct! Guessed: " + str(label))
            counter += 1
        newScore = 100 - (wrong_guess/counter)*100
        allresults.append(newScore)


# EXECUTE TRAINING AND TESTING
# Parameters for test
learningRate = 0.01
momentum_factor = 0.75
epochePerIteration = 1
iterations = 5
trainRunsPerEpoche = -1
testRunsPerEpoche = -1
networkConfig = [784, 600, 100, 10]
nn = NeuralNetwork(networkConfig, learningRate, activation_function=TanhActivationFunction(),
                   momentum_factor=momentum_factor)

# Do training and testing for n iterations
# nn.loadResult()
for i in range(iterations):
    executeEminstTraining(nn, epochePerIteration,
                          trainRunsPerEpoche)
    executeEminstTest(nn, testRunsPerEpoche)
nn.storeResult()
# Plot results
plt.plot(allresults)
plt.ylabel("Test results")
plt.axis([0, iterations-1, 0, 100])
plt.grid()
plt.show()
# Plot all errors
plt.bar(range(10), wrong_guesses, align='center', color="red")
plt.ylabel("Errors per label")
plt.show()
# Store results
text_file = open("Results.txt", "a")
text_file.write(
    f"\n\n Test results for {iterations} iterations with {epochePerIteration} epoches per iteration. There were {trainRunsPerEpoche} train runs and {testRunsPerEpoche} test runs per epoche. Learning rate is {learningRate}. Momentum factor was {str(momentum_factor)}")
text_file.write("\n The network config looked like this: ")
text_file.write(' '.join(str(e) + " " for e in networkConfig))
text_file.write("\n The test results were: ")
text_file.write(' '.join(str(e) + " " for e in allresults))
text_file.write("\n The test errors were: ")
text_file.write(' '.join(str(e) + " " for e in wrong_guesses))
text_file.close()
nn.printSums()
