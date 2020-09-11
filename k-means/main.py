import numpy as np
from collections import Counter
import random

"""
1.Select 'Centroids' (K) initial clusters.
2.Calculate the distance from the 1st point to the 4 initial clusters.
3.Assign the point to the closest cluster.
4.calculate the mean of each cluster.
5.Recluster based on the new means
6.repeat until the clusters no longer change.


How to figure out the amount of Centroids? 
Start at 1, add up 1 after each itteration & look at the variation of the clusters.
"""


def load_data(filename):
    """ Loads a csv file for use by k_nn
      Args:
          filename : name of the file to retrieve the information from
      Returns:
          tuple[list,list]: list of weather measurements and a list of the seasons when the measurements were taken
      """
    data = np.genfromtxt(
        filename,
        delimiter=";",
        usecols=[1, 2, 3, 4, 5, 6, 7],
        converters={
            5: lambda s: 0 if s == b"-1" else float(s),
            7: lambda s: 0 if s == b"-1" else float(s)
        })

    dates = np.genfromtxt(filename, delimiter=";", usecols=[0])
    labels = []
    for label in dates:
        label = str(label)[4:]
        if '0301' <= label < '0601':
            labels.append("lente")
        elif '0601' <= label < '0901':
            labels.append("zomer")
        elif '0901' <= label < '1201':
            labels.append("herfst")
        else:  # from 01-12 to end of year
            labels.append("winter")
    return data, labels

def get_centroid_from_data(data,colNum):
    """
    Returns all the values from a given colums (Vertical)
    """
    setOfPoints = []
    for i in range(len(data)):
        setOfPoints.append(data[i][colNum])
    return setOfPoints

def generate_random_number(max_value):
    """
    Generates a random number with a max value.
    """
    return random.randint(0,max_value)

def place_centroids_at_random(setOfPoints, amountOfCentroids):
    """
    Get X amount of Centroids.
    Return it in a list of positions
    """
    centroid_positions = []
    for _ in range(amountOfCentroids):
        n = generate_random_number(len(setOfPoints))
        #Prevent Duplicates
        while n in centroid_positions:
            n = generate_random_number(len(setOfPoints))
        centroid_positions.append(n)
    return centroid_positions

def k_means(k, data_set):
    centroids = place_centroids_at_random(data_set,k)
    setOfPoints = get_centroid_from_data(data_set,0) #Gets the first Column of data
    return centroids, setOfPoints
    
if __name__ == '__main__':
    # load data from csv
    data_set, data_labels = load_data(r'dataset1.csv')
    k = 5
    k_means(k,data_set)