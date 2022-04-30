import time
import numpy as np


def sigmoid(x):
    """ Returns sigmoid of x. """
    return 1 / (1 + np.exp(-x))


def sigmoid_der(x):
    """ Returns sigmoid derivative of x. """
    return sigmoid(x) * (1 - sigmoid(x))


class Neuron:
    def __init__(self, previous_layer=[]):
        """ Set parameters for Neuron (use default params for input layer). """
        self.previous_layer = previous_layer
        self.weights = np.random.uniform(-1, 1, len(self.previous_layer))
        self.bias    = np.random.uniform(-1, 1)
        self.error   = 0
        self.output  = 0


    def feed_forward(self):
        """ feeds forward weights by calculating neuron_input and output. """
        self.neuron_input = np.sum(np.array([layer.output for layer in self.previous_layer]) * self.weights) + self.bias
        self.output       = sigmoid(self.neuron_input)
        return self.output


    def update(self, learn_rate):
        """ Update weights and bias. """
        self.weights += learn_rate * self.error * np.array([layer.output for layer in self.previous_layer])
        self.bias    += learn_rate * self.error
  
       

class NeuralNetwork:
    def __init__(self, input_layer_size, hidden_layer_size, output_layers_size, learn_rate):
        """ Initializes all layers to train and classify data with. """
        self.learn_rate    = learn_rate
        self.input_layers  = [Neuron(                  ) for _ in range(input_layer_size)  ]
        self.hidden_layers = [Neuron(self.input_layers ) for _ in range(hidden_layer_size) ]
        self.output_layers = [Neuron(self.hidden_layers) for _ in range(output_layers_size)]


    def predict(self, input_data):
        """ Predicts classification of input_data which is a list the length of self.output_layers. """
        for layer, data in zip(self.input_layers, input_data):
            layer.output = data

        for layer in self.hidden_layers:
            layer.feed_forward()

        return [out.feed_forward() for out in self.output_layers]


    def backpropagation(self, desired_output):
        """ Back propagates the error of the hidden and output layers. """
        for i, layer in enumerate(self.hidden_layers):
            error_sum   = sum(output.error * output.weights[i] for output in self.output_layers)
            layer.error = sigmoid_der(layer.neuron_input) * error_sum
        
        for i, layer in enumerate(self.output_layers):
            layer.error = (desired_output[i] - layer.output) * sigmoid_der(layer.neuron_input)


    def update(self):
        """ Updates weights and bias of hidden and output layers. """
        for layer in self.hidden_layers:
            layer.update(self.learn_rate)

        for layer in self.output_layers:
            layer.update(self.learn_rate)


    def train(self, input_data, desired_output, epochs):
        """ Trains layers with input_data and desired_output to get to classify similar data correctly times epochs. """
        for _ in range(epochs):
            for data_in, data_out in zip(input_data, desired_output):
                self.predict(data_in)
                self.backpropagation(data_out)
                self.update()


def get_dataset(filename):
    """ Load data and outputs from file. """
    data    = np.genfromtxt(filename, delimiter=",", usecols=[0, 1, 2, 3])
    outputs = np.genfromtxt(filename, delimiter=",", usecols=[4], dtype=str)
    return data, outputs


def serialize(outputs):
    """ Convert types to list of length of types where correct type is 1 and incorrect is 0. """
    return (np.tile(outputs, (3, 1)).T == np.unique(outputs)).astype(int)


def normalize(data):
    """ Normalize data """
    max_values = np.amax(data, axis=0)
    min_values = np.amin(data, axis=0)
    return (data - min_values) / (max_values - min_values)




if __name__ == "__main__":
    np.random.seed(70)

    data, outputs = get_dataset("iris.data")
    normalized_data = normalize(data)
    serialized_outputs = serialize(outputs)
    

    # learn_rate 0.25600001 and epochs > 5999 = 99.333333333% accurate
    nn = NeuralNetwork(input_layer_size=8, hidden_layer_size=8, output_layers_size=3, learn_rate=0.2)
    nn.train(input_data=normalized_data, desired_output=serialized_outputs, epochs=100)

    print(f"Training took {time.perf_counter():0.2f} seconds")

    correct_predictions = np.all(np.rint([nn.predict(data_point) for data_point in normalized_data]) == serialized_outputs, axis=1)
    print(f"NN was: {np.count_nonzero(correct_predictions) / len(normalized_data) * 100 :0.1f}% correct")

    
    index = 101
    print(f"\nindex {index} is {outputs[index]}")
    prediction = nn.predict(data[index])
    print(f"{prediction[0] * 100 :0.1f}% Iris-Setosa")
    print(f"{prediction[1] * 100 :0.1f}% Iris-Versicolor")
    print(f"{prediction[2] * 100 :0.1f}% Iris-Virginica\n")


