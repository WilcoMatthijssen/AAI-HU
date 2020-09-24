import numpy as np
import random

class Perceptron:
    def __init__(self, train_data, train_labels, epochs, learn_rate):
        self.train_data = train_data
        self.train_labels = train_labels
        self.epochs = epochs
        self.learn_rate = learn_rate

        self.accuracy = 0
        self.samples = 0

        self.bias = 0
        self.weights = np.array([random.uniform(-1, 1) for _ in range(len(train_data[0]))])

    def curr_accuracy(self):
        return round(self.accuracy/self.samples, 3)

    def activation(self, n):
        print(n)
        return 1 / (1 + np.exp(-n))
        return 0 if n < 0 else 1

    def predict(self, data):
        # total = sum(data.dot(self.weights)) + self.bias

        total = self.bias

        for data_val, weight in zip(data, self.weights):
            total += data_val * weight
        return self.activation(total)

    def train(self):
        for _ in range(self.epochs):
            for train_sample, train_label in zip(self.train_data, self.train_labels):
                prediciton = self.predict(train_sample)

                #print("prediction is {} and actual label is {}".format(prediciton, train_label))
                self.accuracy += +1 if prediciton == train_label else -1
                self.samples += 1

                loss = train_label - prediciton


                for i in range(len(self.weights)):
                    self.weights[i] += loss * train_sample[i] * self.learn_rate
                # for weight, sample_val in zip(self.weights, train_sample):
                #     weight += loss * sample_val * self.learn_rate

                self.bias += loss * self.learn_rate
            print("accuracy", self.curr_accuracy())


def data_from_csv_file(filename):
    data = np.genfromtxt(filename,delimiter=',', dtype=float,usecols=(0,1,2,3))
    str_labels = np.genfromtxt(filename,delimiter=',', dtype=str, usecols=(4))

    # convert labels to 0 or 1
    labels = [1 if str_label == 'Iris-versicolor' else 0 for str_label in str_labels]

    # labels= []
    # for str_label in str_labels:
    #     if str_label == 'Iris-versicolor':
    #         labels.append(0)
    #     elif str_label == "Iris-virginica":
    #         labels.append(1)
    #     else:
    #         labels.append(2)

    return data, labels

def normalize(data):
    max_values = [np.max(column) for column in data.T]
    min_values = [np.min(column) for column in data.T]
    for feature in data:
        for i in range(len(feature)):
            feature[i] = (feature[i] - min_values[i]) / (max_values[i] - min_values[i])

if __name__ == "__main__":
    data, labels = data_from_csv_file("iris.data")
    normalize(data)
    p = Perceptron(data, labels, 1000, 1)
    p.train()

    # "Iris-setosa" vs "Iris-versicolor"
    #
    # "Iris-setosa" vs "Iris-virginica"
    #
    # "Iris-versicolor" vs  "Iris-virginica"

