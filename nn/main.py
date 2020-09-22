import numpy as np
from numpy import genfromtxt
import csv
import math

def assignment4_1():
    """
    NOR GATE
    3 INPUTS
    1 OUTPUT 
    THRESHOLD 
    """
    sig = lambda x: 1/(1+math.exp(-x))
    w = np.array([1,1,1])

    def OR(x,w,act):
        return act(w.dot(x))

    x = np.array([1,0,0])
    return not bool(round(OR(x,w,sig)))

def data_to_csv_file(filename):
    return genfromtxt(filename,dtype=float,delimiter=',',usecols=(0,1,2,3))
    
def normalize(data):
    max_values = [np.max(column) for column in data.T]
    min_values = [np.min(column) for column in data.T]
    for feature in data:
        for i in range(len(feature)):
            feature[i] = (feature[i] - min_values[i]) / (max_values[i] - min_values[i])

    
class Neuron:
    def ___init____(self):
        self.name = "Neuron"
         
class Perceptron:
    def ___init____(self):
        pass
    
class Adder:
    def ___init____(self):
        self.name = "Adder"

if __name__ == "__main__":
    data = data_to_csv_file("iris.data")
    normalize(data)
    print(assignment4_1())


