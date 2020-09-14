import numpy as np
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
    """ loads a csv into an dataset and datalabel list """


def get_distance(point_a, point_b):
    """ Calculates distance between point_a and point_b """


def get_nearest_centroid_index(data_point, centroids):
    """ Gets the index of the centroid that is closest to data_point with the use of get_distance. """


def cluster_data(data_points, centroids):
    """ Clusters data_points in data_points together that have the same centroid """


def get_k_random_centroids(k, data_points):
    """ Get k centroids from random unique elements in data_points """


def get_centroids_from_clusters(clusters):
    """ Gets new centroids from the mean/average of each cluster in data_clusters """


def k_means(k, data_points, max_iters=100):
    """ Returns data_points spread over k clusters """


def cluster_data_into_seasons(clusters, data_points, data_labels):
    """ Maximum vote principle to cluster the data into the 4 different seasons """


if __name__ == '__main__':
    # volgens inlever pdf moet je bij random een seed gebruiken
    random.seed(69)

    # load data from csv
    data_points, data_labels = load_data(r'validation1.csv')

    # Result of k_means
    result = k_means(k=4, data_points=data_points)

    # Maximum vote principle to cluster the data into the 4 different seasons
    season_clusters = cluster_data_into_seasons(result, data_points, data_labels)
    for season in season_clusters:
        print(season)

    #np.set_printoptions(threshold=np.inf)

    diff = {}
    for k in range(2,10):
        clusters = k_means(k, data_points)
        cluster_sizes = [len(cluster) for cluster in clusters]
        print(k)
        diff["k{}".format(k)] = np.std(cluster_sizes)

    print(diff)


