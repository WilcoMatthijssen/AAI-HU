import numpy as np
from numpy import genfromtxt
import csv
import math

def assignment4_1():
    """
    NOR GATE
    3 INPUTS 0.1
    1 OUTPUT 0/1
    THRESHOLD 0.1
    """
    sig = lambda x: 1/(1+math.exp(-x))
    w = np.array([15,-10,-10])
    def NOT(x,w,act):
        return (act(w.dot(x)))
    x = np.array([1,0,1])
    return NOT(x,w,sig)

def gib_data():
    def data_to_csv_file(filename):
        return genfromtxt(filename,dtype=float,delimiter=',',usecols=(0,1,2,3))
        
    def normalize(data):
        max_values = [np.max(column) for column in data.T]
        min_values = [np.min(column) for column in data.T]
        for feature in data:
            for i in range(len(feature)):
                feature[i] = (feature[i] - min_values[i]) / (max_values[i] - min_values[i])
    
    data = data_to_csv_file("iris.data")
    normalize(data)
    return data
    
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
    a = gib_data()
    print(a)


