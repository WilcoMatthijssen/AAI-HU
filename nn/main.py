import time
import numpy as np
import random


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_der(x):
    return sigmoid(x) * (1 - sigmoid(x))


class Neuron:

    def __init__(self, is_sigmoid=True, weights=None, threshold=None, bias=None, previous_layer=None):
        """ Set parameters for Neuron (use default params for input layer). """
        self.previous_layer = previous_layer
        self.weights = weights
        self.bias = bias
        self.threshold = threshold
        self.is_sigmoid = is_sigmoid

        self.error = 0
        self.output = 0
        self.neuron_input = 0


    def feed_forward(self):
        self.neuron_input = sum(val.output * w for val, w in zip(self.previous_layer, self.weights))
        if self.is_sigmoid:
            self.neuron_input += self.bias
            self.output = sigmoid(self.neuron_input)
        else:
            self.output = 1 if self.neuron_input >= self.threshold else 0
        return self.output

    def update(self, learn_rate):
        for i in range(len(self.weights)):
            self.weights[i] += learn_rate * self.error * self.previous_layer[i].output

        self.bias += learn_rate * self.error


class NeuralNetwork:

    def __init__(self, input_layer_size, output_layers_size, is_sigmoid, learn_rate):
        self.learn_rate = learn_rate

        self.input_layers = [Neuron() for _ in range(input_layer_size)]

        self.hidden_layers = []
        for i in range(input_layer_size):
            weights = [random.uniform(-1, 1) for _ in range(input_layer_size)]
            self.hidden_layers.append(
                Neuron(is_sigmoid, weights, random.uniform(-1, 1), random.uniform(-1, 1), self.input_layers))

        self.output_layers = []
        for i in range(output_layers_size):
            weights = [random.uniform(-1, 1) for _ in range(input_layer_size)]
            self.output_layers.append(
                Neuron(is_sigmoid, weights, random.uniform(-1, 1), random.uniform(-1, 1), self.hidden_layers))

    def predict(self, input_data):
        for input_l, input_d in zip(self.input_layers, input_data):
            input_l.output = input_d

        for hidden in self.hidden_layers:
            hidden.feed_forward()

        return [out.feed_forward() for out in self.output_layers]

    def backpropagation(self, desired_output):
        for i in range(len(self.output_layers)):
            self.output_layers[i].error = (desired_output[i] - self.output_layers[i].output) * sigmoid_der(self.output_layers[i].neuron_input)

        for i in range(len(self.hidden_layers)):
            error_sum = sum(output.error * output.weights[i] for output in self.output_layers)
            self.hidden_layers[i].error = sigmoid_der(self.hidden_layers[i].neuron_input) * error_sum

    def update(self):
        for o_layer in self.output_layers:
            o_layer.update(self.learn_rate)

        for h_layer in self.hidden_layers:
            h_layer.update(self.learn_rate)

    def train(self, input_data, desired_output, epochs):
        for _ in range(epochs):
            for data_in, data_out in zip(input_data, desired_output):
                self.predict(data_in)
                self.backpropagation(data_out)
                self.update()


def get_dataset(filename):
    data = np.genfromtxt(filename, delimiter=",", usecols=[0, 1, 2, 3])
    outputs = np.genfromtxt(filename, delimiter=",", usecols=[4], dtype=str)
    return data, outputs


def serialize_outputs(outputs):
    serialized_outputs = []

    for data_output in outputs:
        if data_output == "Iris-setosa":
            serialized_outputs.append(np.array([1, 0, 0]))
        if data_output == "Iris-versicolor":
            serialized_outputs.append(np.array([0, 1, 0]))
        if data_output == "Iris-virginica":
            serialized_outputs.append(np.array([0, 0, 1]))
    return np.array(serialized_outputs)


if __name__ == "__main__":
    random.seed(0)
    data, outputs = get_dataset("iris.data")

    serialized_outputs = serialize_outputs(outputs)

    nn = NeuralNetwork(4, 3, True, 0.1)
    nn.train(data, serialized_outputs, 350)

    print(f"Training took {time.perf_counter():0.2f} seconds")

    guess = np.array([np.argmax(nn.predict(data_point)) for data_point in data])
    actual = np.array([np.argmax(output) for output in serialized_outputs])
    print("NN was: ", (np.count_nonzero(guess == actual) / len(data) * 100), "% correct")

    index = 40
    print(f"\nindex {index} is {outputs[index]}")
    tempNetworkOutput = nn.predict(data[index])
    print(round(tempNetworkOutput[0], 2) * 100, "% Iris-Setosa")
    print(round(tempNetworkOutput[1], 2) * 100, "% Iris-Versicolor")
    print(round(tempNetworkOutput[2], 2) * 100, "% Iris-Virginica\n")

