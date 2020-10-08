import time
import numpy as np
import random


def sigmoid(x):
    """ Returns sigmoid of x. """
    return 1 / (1 + np.exp(-x))


def sigmoid_der(x):
    """ Returns sigmoid derivative of x. """
    return sigmoid(x) * (1 - sigmoid(x))


class Neuron:

    def __init__(self, weights=None, previous_layer=None):
        """ Set parameters for Neuron (use default params for input layer). """
        self.previous_layer = previous_layer
        self.weights = weights
        self.bias = random.uniform(-1, 1)
        self.threshold = random.uniform(-1, 1)

        self.error = 0
        self.output = 0
        self.neuron_input = 0

    def feed_forward(self):
        """ feeds forward weights by calculating neuron_input and output. """
        self.neuron_input = sum(val.output * w for val, w in zip(self.previous_layer, self.weights))
        self.neuron_input += self.bias
        self.output = sigmoid(self.neuron_input)
        return self.output

    def update(self, learn_rate):
        """ Update weights and bias. """
        for i in range(len(self.weights)):
            self.weights[i] += learn_rate * self.error * self.previous_layer[i].output

        self.bias += learn_rate * self.error


class NeuralNetwork:

    def __init__(self, input_layer_size, output_layers_size, learn_rate):
        """ Initializes all layers to train and classify data with. """
        self.learn_rate = learn_rate

        self.input_layers = [Neuron() for _ in range(input_layer_size)]

        self.hidden_layers = []
        for i in range(input_layer_size):
            weights = [random.uniform(-1, 1) for _ in range(input_layer_size)]
            self.hidden_layers.append(
                Neuron(weights, self.input_layers))

        self.output_layers = []
        for i in range(output_layers_size):
            weights = [random.uniform(-1, 1) for _ in range(input_layer_size)]
            self.output_layers.append(
                Neuron(weights, self.hidden_layers))

    def predict(self, input_data):
        """ Predicts classification of input_data which is a list the length of self.output_layers. """
        for layer, data in zip(self.input_layers, input_data):
            layer.output = data

        for layer in self.hidden_layers:
            layer.feed_forward()

        return [out.feed_forward() for out in self.output_layers]

    def backpropagation(self, desired_output):
        """ Back propagates the error of the hidden and output layers. """
        for i in range(len(self.output_layers)):
            self.output_layers[i].error = (desired_output[i] - self.output_layers[i].output) * sigmoid_der(self.output_layers[i].neuron_input)

        for i in range(len(self.hidden_layers)):
            error_sum = sum(output.error * output.weights[i] for output in self.output_layers)
            self.hidden_layers[i].error = sigmoid_der(self.hidden_layers[i].neuron_input) * error_sum

    def update(self):
        """ Updates weights and bias of hidden and output layers. """
        for o_layer in self.output_layers:
            o_layer.update(self.learn_rate)

        for h_layer in self.hidden_layers:
            h_layer.update(self.learn_rate)

    def train(self, input_data, desired_output, epochs):
        """ Trains layers with input_data and desired_output to get to classify similar data correctly times epochs. """
        for _ in range(epochs):
            for data_in, data_out in zip(input_data, desired_output):
                self.predict(data_in)
                self.backpropagation(data_out)
                self.update()


def get_dataset(filename):
    """ Load data and outputs from file. """
    data = np.genfromtxt(filename, delimiter=",", usecols=[0, 1, 2, 3])
    outputs = np.genfromtxt(filename, delimiter=",", usecols=[4], dtype=str)
    return data, outputs


def serialize_outputs(outputs):
    """ Convert types to list of length of types where correct type is 1 and incorrect is 0. """
    return [np.array(np.unique(outputs) == name).astype(int) for name in outputs]

def normalize(data):
    """ Normalize data """
    max_values = [np.max(column) for column in data.T]
    min_values = [np.min(column) for column in data.T]
    for feature in data:
        for i in range(len(feature)):
            feature[i] = (feature[i] - min_values[i]) / (max_values[i] - min_values[i])

if __name__ == "__main__":
    random.seed(0)

    data, outputs = get_dataset("iris.data")

    serialized_outputs = serialize_outputs(outputs)

    normalize(data)

    # learn_rate 0.25600001 and epochs > 5999 = 99.333333333% accurate
    nn = NeuralNetwork(input_layer_size=4, output_layers_size=3, learn_rate=0.2)
    nn.train(input_data=data, desired_output=serialized_outputs, epochs=100)

    print(f"Training took {time.perf_counter():0.2f} seconds")

    guess = np.array([np.argmax(nn.predict(data_point)) for data_point in data])
    actual = np.array([np.argmax(output) for output in serialized_outputs])
    print("NN was: ", (np.count_nonzero(guess == actual) / len(data) * 100), "% correct")

    index = 101
    print(f"\nindex {index} is {outputs[index]}")
    prediction = nn.predict(data[index])
    print(round(prediction[0], 2) * 100, "% Iris-Setosa")
    print(round(prediction[1], 2) * 100, "% Iris-Versicolor")
    print(round(prediction[2], 2) * 100, "% Iris-Virginica\n")


