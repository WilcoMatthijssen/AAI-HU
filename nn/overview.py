import time
import numpy as np
import random


def sigmoid(x):
    """ Returns sigmoid of x. """



def sigmoid_der(x):
    """ Returns sigmoid derivative of x. """



class Neuron:

    def __init__(self, weights=None, previous_layer=None):
        """ Set parameters for Neuron (use default params for input layer). """

    def feed_forward(self):
        """ feeds forward weights by calculating neuron_input and output. """

    def update(self, learn_rate):
        """ Update weights and bias. """



class NeuralNetwork:

    def __init__(self, input_layer_size, output_layers_size, learn_rate):
        """ Initializes all layers to train and classify data with. """


    def predict(self, input_data):
        """ Predicts classification of input_data which is a list the length of self.output_layers. """


    def backpropagation(self, desired_output):
        """ Back propagates the error of the hidden and output layers. """


    def update(self):
        """ Updates weights and bias of hidden and output layers. """


    def train(self, input_data, desired_output, epochs):
        """ Trains layers with input_data and desired_output to get to classify similar data correctly times epochs. """


def get_dataset(filename):
    """ Load data and outputs from file. """


def serialize_outputs(outputs):
    """ Convert types to list of length of types where correct type is 1 and incorrect is 0. """


def normalize(data):
    """ Normalize data """


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


