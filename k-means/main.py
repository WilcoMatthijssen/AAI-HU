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
    Returns all the values from a given columns (Vertical)
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



def get_distance(point_a, point_b):
    """ Calculates distance between point_a and point_b """
    distance = 0.0
    for ax, bx in zip(point_a, point_b):
        distance += pow(ax - bx, 2)
    return np.sqrt(distance)

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



def find_nearest_centroid_index(data_point, centroids):
    """ Finds the index of nearest centroid with the use of get_distance. """
    # maak list met alle afstanden tussen het data_point en centroids
    distances = [get_distance(data_point, centroid) for centroid in centroids]

    # pak index van laagste waarde/afstand
    return np.argmin(distances)


def cluster_data(data_set, centroids):
    """ Cluster to each data_point in data_set to the nearest centroid """
    clustered_data = [[] for _ in range(len(centroids))]
    for data_point in data_set:
        nearest_centroid_index = find_nearest_centroid_index(data_point, centroids)
        clustered_data[nearest_centroid_index].append(data_point)
    return clustered_data


def get_centroids_from_random(amount_of_centroids, data_set):
    """ Get amount_of_centroids centroids from random elements in data_set """
    centroids = []
    for _ in range(amount_of_centroids):
        centroid = [random.choice(column) for column in data_set.T]
        centroids.append(centroid)
    return np.array(centroids)


def get_centroids_from_clusters(data_clusters):
    """ Gets the mean/average from each cluster in data_clusters """
    centroids = []
    for cluster in data_clusters:
        if len(cluster) == 0:
            centroids.append(np.array([np.NINF for _ in range(len(data_clusters[0]))]))
        else:
            centroids.append(np.mean(cluster, axis=0))
    return np.array(centroids)


def k_means(k, data_set):
    """ Finds the centroid for k clusters """
    # centroids = place_centroids_at_random(data_set,k)
    # setOfPoints = get_centroid_from_data(data_set,0) #Gets the first Column of data
    # return centroids, setOfPoints

    centroids = get_centroids_from_random(k, data_set)
    while True:
        clusters = cluster_data(data_set, centroids)
        new_centroids = get_centroids_from_clusters(clusters)
        if (centroids == new_centroids).all():
            centroids = new_centroids
            break
        else:
            centroids = new_centroids
    return cluster_data(data_set, centroids)


def cluster_data_into_seasons(clusters, data_set, data_labels):
    """ Maximum vote principle to cluster the data into the 4 different seasons """
    season_cluster = []
    for cluster in clusters:
        collection_count = {}
        for element in cluster:
            for set_elem, label in zip(data_set, data_labels):
                if (set_elem == element).all():
                    if label not in collection_count:
                        collection_count[label] = 0
                    collection_count[label] += 1
                    break
        season = max(collection_count, key=collection_count.get)
        season_cluster.append([season, cluster])
    return season_cluster
    
if __name__ == '__main__':
    # volgens inlever pdf moet je bij random een seed gebruiken
    random.seed(69)

    # load data from csv
    data_set, data_labels = load_data(r'dataset1.csv')

    # Result of k_means
    result = k_means(k=4, data_set=data_set)

    # Maximum vote principle to cluster the data into the 4 different seasons
    season_clusters = cluster_data_into_seasons(result, data_set, data_labels)

    # for season in season_clusters:
    #     print(season)

    #np.set_printoptions(threshold=np.inf)

    diff = {}
    for k in range(1,6):
        clusters = k_means(k, data_set)
        cluster_sizes = [len(cluster) for cluster in clusters]
        diff["k{}".format(k)] = max(cluster_sizes) - min(cluster_sizes)

    print(diff)










