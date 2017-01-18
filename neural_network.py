import sys
import numpy as np
import pdb
import math
import random

quadraticCost = False

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def costFunction(outputActivations, label):
    if quadraticCost:
        # (a - y) at the index where it's classified
        copy = np.copy(outputActivations)
        copy[int(label)] -= 1
        return copy
    else:
        copy = np.copy(outputActivations)
        vLabel = toVector(label)
        return (-1.0) * ( (vLabel / (1.0 *copy)) + (vLabel - 1.0) * (1.0 / 1.0 - copy) )

def toVector(y):
    vLabel = np.zeros((10,1))
    vLabel[int(y)] = 1.0
    return vLabel

class NeuralNetwork:
    def __init__(self, nI = 784, nH = 30, nO = 10, epochs = 30, learnRate = 3, miniBatchSize = 20, isOriginal = False):
        self.nInput = nI
        self.nHidden = nH
        self.nOutput = nO

        self.epochs = epochs
        self.learnRate = learnRate
        self.miniBatchSize = miniBatchSize
        self.isOriginal = isOriginal

        if isOriginal:
            self.biases = [ np.array([1,1]).reshape((2,1)) , np.array([1,1]).reshape((2,1)) ]
            weights = []
            weights.append(np.array([ [0.1, 0.1] , [0.2, 0.1] ]))
            weights.append(np.array([ [0.1, 0.1] , [0.1, 0.2] ]))
            self.weights = weights
        else:
            # Setting initial random w and b values
            biases = [np.random.randn(nH, 1) , np.random.randn(nO, 1)]
            # weights = [np.random.randn(nH, nI), np.random.randn(nO, nH)]
            self.biases = biases
            weights = [np.random.randn(nH, nI), np.random.randn(nO, nH)]
            # weights = [np.random.randn(nH, nI), np.random.randn(nO, nH)]
            self.weights = weights

    def feedForward(self, features):
        features = np.array(features).reshape((len(features), 1))
        # middleActivations = (30x1)
        middleActivations = sigmoid(np.dot(self.weights[0], features) + self.biases[0])

        # Feed forward to output layer
        # outputActivations = (10x1)
        outputActivations = sigmoid(np.dot(self.weights[1], middleActivations) + self.biases[1])

        # return the output activations array
        return outputActivations

    def processMiniBatch(self, miniBatch):
        errb = [np.zeros(bLayer.shape) for bLayer in self.biases]
        errw = [np.zeros(wLayer.shape) for wLayer in self.weights]

        for x, y in miniBatch:
            # Run backprop algo
            newErrb, newErrw = self.bprop(x, y)
            # You should now have errors for all biases and weights. They are also summed already (but not averaged!)

            for layer, errors in enumerate(newErrb):
                errb[layer] = errb[layer] + errors
            for layer, errors in enumerate(newErrw):
                errw[layer] = errw[layer] + errors

        # Update weights and biases
        factor = (float(self.learnRate) / self.miniBatchSize)

        for ind, arr in enumerate(self.weights):
            # each of self.weights[0][ind] = (784,)
            self.weights[ind] = arr - (factor * errw[ind])

        # Hidden biases:
        self.biases[0] = self.biases[0] - (factor * errb[0])
        # Output biases:
        self.biases[1] = self.biases[1] - (factor * errb[1])

    def bprop(self, features, label):

        # NOTE NOTE START: FEED FORWARD
        # features = (784x1)
        features = np.array(features).reshape((len(features), 1))
        activations = [features]
        # middleActivations = (30x1)
        middleActivations = sigmoid(np.dot(self.weights[0], features) + self.biases[0])
        activations.append(middleActivations)

        # Feed forward to output layer
        # outputActivations = (10x1)
        outputActivations = sigmoid(np.dot(self.weights[1], middleActivations) + self.biases[1])
        activations.append(outputActivations)
        # NOTE NOTE END: FEED FORWARD

        # dCostActivation = (10x1)
        dCostActivation = costFunction(outputActivations, label)
        # outputDADG = (10x1)
        outputDADG = outputActivations * (1 - outputActivations)
        # outputErrbs = (10x1) x (10x1)
        outputErrbs = outputDADG * dCostActivation

        # NOTE NOTE NOTE NOTE NOTE NOTE NOTE NOTE NOTE NOTE NOTE NOTE NOTE NOTE

        # outputErrws = (10x30)
        outputErrws = np.zeros(self.weights[1].shape)
        # calculate weights from hidden-output
        for i in range(0, self.nOutput):
            # Output layer errb
            outputErrb = outputErrbs[i]
            # (30x1)
            oErrw = middleActivations * outputErrb
            outputErrws[i] = oErrw.reshape((len(oErrw),))

        # NOTE NOTE NOTE NOTE NOTE NOTE NOTE NOTE NOTE NOTE NOTE NOTE NOTE NOTE

        # Calculate hidden layer ERRBs
        hiddenErrbs = []
        hiddenDADG = middleActivations * (1 - middleActivations)

        for num in range(0, self.nHidden):
            weights = []

            # Get weights connected to b_1i
            for i in range(0, self.nOutput):
                weights.append(self.weights[1][i][num])

            err = np.dot(weights, outputErrbs) * hiddenDADG[num]
            hiddenErrbs.append(err)
        hiddenErrbs = np.array(hiddenErrbs).reshape((self.nHidden, 1))

        # NOTE NOTE NOTE NOTE NOTE NOTE NOTE NOTE NOTE NOTE NOTE NOTE NOTE NOTE

        # Calculate input-hidden ERRWs

        # hiddenErrws = (30x784)
        hiddenErrws = np.zeros(self.weights[0].shape)
        # calculate weights from input-hidden
        for i in range(0, self.nHidden):
            # Hidden layer errb
            hiddenErrb = hiddenErrbs[i]
            hErrw = features * hiddenErrb
            hiddenErrws[i] = hErrw.reshape((len(hErrw),))

        return [hiddenErrbs, outputErrbs], [hiddenErrws, outputErrws]

    def train(self, trainData, testData = None):
        for epNum in range(1, self.epochs + 1):
            random.shuffle(trainData)

            batches = []
            for idx in range(0, len(trainData), self.miniBatchSize):
                batches.append(trainData[idx : idx + self.miniBatchSize])

            numBatches = len(batches)
            for miniBatch in batches:
                self.processMiniBatch(miniBatch)

            if self.isOriginal:
                print(self.weights)
                print(self.biases)

            if testData:
                self.evaluate(testData)

            print "Epoch {num} complete.".format(num= epNum)

    def evaluate(self, testData):
        correct = 0

        for testExample in testData:
            features = testExample[0]
            label = int(testExample[1])
            outputActivations = self.feedForward(features)

            if np.argmax(outputActivations) == label:
                correct += 1

        print "Accuracy: {correct} / {total}".format(correct=correct, total = len(testData))

    def makePredictions(self, testX, outputName):
        if outputName:
            predictions = [np.argmax(self.feedForward(features)) for features in testX]
            np.savetxt(outputName, predictions, fmt='%i', delimiter=',')

######### MAIN CODE #########

print('Reading inputs...')

if sys.argv[1] == "original":
    nInput = 2
    nHidden = 2
    nOutput = 2
    epochs = 3
    learnRate = 0.1
    miniBatchSize = 2

    trainX = [ np.array([0.1, 0.1]) , np.array([0.1, 0.2]) ]
    trainY = [ 0 , 1 ]
    train = zip(trainX, trainY)

    nn = NeuralNetwork(nInput, nHidden, nOutput, epochs, learnRate, miniBatchSize, True)

    nn.train(train)
else:
    nInput = int(sys.argv[1])
    nHidden = int(sys.argv[2])
    nOutput = int(sys.argv[3])
    trainX = np.loadtxt(sys.argv[4], delimiter=',')
    trainY = np.loadtxt(sys.argv[5], delimiter=',')
    testX = np.loadtxt(sys.argv[6], delimiter=',')
    testY = np.loadtxt('TestDigitY.csv.gz', delimiter=',')
    predictY = sys.argv[7] if len(sys.argv) > 7 else None

    # Defaults
    epochs = 30
    learnRate = 3
    miniBatchSize = 20

    train = zip(trainX, trainY)
    test = zip(testX, testY)
    print('...done.')

    nn = NeuralNetwork(nInput, nHidden, nOutput, epochs, learnRate, miniBatchSize)
    nn.train(train, test)
    nn.makePredictions(testX, predictY)
