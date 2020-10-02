import time
import numpy as np
import random


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_der(x):
    return sigmoid(x) * (1 - sigmoid(x))


class NeuralNetwork:

    def __init__(self, input_layer_size, output_layers_size, isSigmoid=True, learningRate=0.1):
        self.learnRate = learningRate

        self.input_layers = [Neuron() for _ in range(input_layer_size)]

        self.hidden_layers = []
        for i in range(input_layer_size):
            weights = [random.uniform(-1, 1) for _ in range(input_layer_size)]
            self.hidden_layers.append(
                Neuron(isSigmoid, weights, random.uniform(-1, 1), random.uniform(-1, 1), self.input_layers))

        self.output_layers = []
        for i in range(output_layers_size):
            weights = [random.uniform(-1, 1) for _ in range(input_layer_size)]
            self.output_layers.append(
                Neuron(isSigmoid, weights, random.uniform(-1, 1), random.uniform(-1, 1), self.hidden_layers))

    def predict(self, input_data):
        for input_l, input_d in zip(self.input_layers, input_data):
            input_l.output = input_d

        for hidden in self.hidden_layers:
            hidden.feedForward()

        return [out.feedForward() for out in self.output_layers]

    def backpropagation(self, desired_output):
        for i in range(len(self.output_layers)):
            self.output_layers[i].error = (desired_output[i] - self.output_layers[i].output) * sigmoid_der(self.output_layers[i].neuronInput)

        for i in range(len(self.hidden_layers)):
            error_sum = sum(output.error * output.weights[i] for output in self.output_layers)
            self.hidden_layers[i].error = sigmoid_der(self.hidden_layers[i].neuronInput) * error_sum

    def update(self):
        for o_layer in self.output_layers:
            o_layer.update(self.learnRate)

        for h_layer in self.hidden_layers:
            h_layer.update(self.learnRate)

    def train(self, input_data, desired_output, epochs):
        for _ in range(epochs):
            for data_in, data_out in zip(input_data, desired_output):
                self.predict(data_in)
                self.backpropagation(data_out)
                self.update()


class Neuron:

    def __init__(self, isSigmoid=True, weights=None, threshold=None, bias=None, previous_layer=None):
        self.previous_layer = previous_layer
        self.weights = weights

        self.error = 0
        self.bias = bias
        self.threshold = threshold
        self.output = 0
        self.neuronInput = 0
        self.isSigmoid = isSigmoid

    def feedForward(self):
        self.neuronInput = sum(val.output * w for val, w in zip(self.previous_layer, self.weights))
        if self.isSigmoid:
            self.neuronInput += self.bias
            self.output = sigmoid(self.neuronInput)
        else:
            self.output = 1 if self.neuronInput >= self.threshold else 0
        return self.output

    def update(self, learn_rate):
        for i in range(len(self.weights)):
            self.weights[i] += learn_rate * self.error * self.previous_layer[i].output

        self.bias += learn_rate * self.error


if __name__ == "__main__":
    random.seed(0)
    trainingData = np.genfromtxt("iris.data", delimiter=",", usecols=[0, 1, 2, 3])
    trainingDataOutput = np.genfromtxt("iris.data", delimiter=",", usecols=[4], dtype=str)

    convertedOutput = []

    for i in range(len(trainingDataOutput)):
        if trainingDataOutput[i] == "Iris-setosa":
            convertedOutput.append([1, 0, 0])
        if trainingDataOutput[i] == "Iris-versicolor":
            convertedOutput.append([0, 1, 0])
        if trainingDataOutput[i] == "Iris-virginica":
            convertedOutput.append([0, 0, 1])

    irisNetwork = NeuralNetwork(4, 3, True, 0.05)
    irisNetwork.train(trainingData, convertedOutput, 1000)
    runtime = time.perf_counter()
    print(f"Training took {runtime:0.4f} seconds")

    correctCounter = 0
    for i in range(len(trainingData)):
        tempOutput = irisNetwork.predict(trainingData[i])
        correctCounter += convertedOutput[i][tempOutput.index(max(tempOutput))]


    print("NN was: ", (correctCounter / len(trainingData) * 100), "% correct")

    dataSetIndex = 40

    print(f"\nNN predicts index {dataSetIndex} is {trainingDataOutput[dataSetIndex]}")

    tempNetworkOutput = irisNetwork.predict(trainingData[dataSetIndex])
    print(round(tempNetworkOutput[0], 2) * 100,"% Iris-Setosa")
    print(round(tempNetworkOutput[1], 2) * 100,"% Iris-Versicolor")
    print(round(tempNetworkOutput[2], 2) * 100,"% Iris-Virginica\n")