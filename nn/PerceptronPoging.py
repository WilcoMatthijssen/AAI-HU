import numpy as np
import random

class Perceptron:
    def __init__(self, learn_data, learn_labels, epochs, learn_rate):
        self.learn_data = learn_data
        self.learn_labels = learn_labels
        self.epochs = epochs
        self.learn_rate = learn_rate

        self.bias = 0.00
        self.weights = np.array([random.uniform(-1, 1) for _ in range(len(learn_data[0]))])
        #self.weights = np.zeros(len(train_data[0]))

    def predict(self, item):
        total = np.dot(item, self.weights) + self.bias
        #return 1 if total > 0 else 0
        return 1/(1 + np.exp(-total))

    def train(self):
        for _ in range(self.epochs):
            for item, target in zip(self.learn_data, self.learn_labels):

                prediction = self.predict(item)
                update = (target - prediction) * self.learn_rate
                self.bias += update
                self.weights += update * item



def data_from_csv_file(filename):
    data = np.genfromtxt(filename,delimiter=',', dtype=float,usecols=(0,1,2,3))
    labels = np.genfromtxt(filename,delimiter=',', dtype=str, usecols=(4))

    return data, labels

def normalize(data):
    max_values = [np.max(column) for column in data.T]
    min_values = [np.min(column) for column in data.T]
    for feature in data:
        for i in range(len(feature)):
            feature[i] = (feature[i] - min_values[i]) / (max_values[i] - min_values[i])


class NN:
    def __init__(self, data, labels, epochs, learning_rate):
        self.perceptrons = []
        self.unique_labels = list(set(labels))

        for unique_label in self.unique_labels:
            l = [1 if unique_label == label else 0 for label in labels]
            self.perceptrons.append(Perceptron(data, l, epochs, learning_rate))


    def train(self):
        for perceptron in self.perceptrons:
            perceptron.train()

    def predict(self, data):
        predictions = [p.predict(data) for p in self.perceptrons]
        # type_prediction = self.unique_labels[np.argmax(predictions)]
        # return type_prediction
        return predictions



if __name__ == "__main__":
    data, labels = data_from_csv_file("iris.data")
    normalize(data)

    n = NN(data, labels, 100, 0.1)
    n.train()

    print(n.unique_labels)
    print(n.predict(data[110]))



