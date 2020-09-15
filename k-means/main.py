import numpy as np
import random
import matplotlib.pyplot as plt

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


def get_distance(point_a, point_b):
    """ Calculates distance between point_a and point_b """
    distance = 0.0
    for ax, bx in zip(point_a, point_b):
        distance += pow(ax - bx, 2)
    return np.sqrt(distance)


def get_nearest_centroid_index(data_point, centroids):
    """ Gets the index of the centroid that is closest to data_point with the use of get_distance. """
    # maak list met alle afstanden tussen het data_point en centroids
    distances = [get_distance(data_point, centroid) for centroid in centroids]

    # pak index van laagste waarde/afstand
    return np.argmin(distances)


def cluster_data(data_points, centroids):
    """ Clusters data_points in data_points together that have the same centroid """
    clustered_data = [[] for _ in range(len(centroids))]
    for data_point in data_points:
        nearest_centroid_index = get_nearest_centroid_index(data_point, centroids)
        clustered_data[nearest_centroid_index].append(data_point)
    return clustered_data


def get_k_random_centroids(k, data_points):
    """ Get k centroids from random unique elements in data_points """
    return np.array(random.sample(list(data_points), k))


def get_centroids_from_clusters(clusters):
    """ Gets new centroids from the mean/average of each cluster in data_clusters """
    return np.array([np.mean(cluster, axis=0) for cluster in clusters])


def k_means(k, data_points, max_iters=100):
    """ Returns data_points spread over k clusters """
    centroids = get_k_random_centroids(k, data_points)
    for _ in range(max_iters):
        clusters = cluster_data(data_points, centroids)
        new_centroids = get_centroids_from_clusters(clusters)
        if (centroids == new_centroids).all():
            return clusters
        else:
            centroids = new_centroids


def cluster_data_into_seasons(clusters, data_points, data_labels):
    """ Maximum vote principle to cluster the data into the 4 different seasons """
    season_cluster = []
    for cluster in clusters:
        collection_count = {}
        for element in cluster:
            for set_elem, label in zip(data_points, data_labels):
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
    data_points, data_labels = load_data(r'k-means\validation1.csv')

    # Result of k_means
    result = k_means(k=4, data_points=data_points)

    # Maximum vote principle to cluster the data into the 4 different seasons
    season_clusters = cluster_data_into_seasons(result, data_points, data_labels)
    # for season in season_clusters:
    #     print(season)

    #np.set_printoptions(threshold=np.inf)

    diff = {}
    for k in range(2,100):
        clusters = k_means(k, data_points)
        cluster_sizes = [len(cluster) for cluster in clusters]
        diff[k] = np.std(cluster_sizes)

    #Scree Plot
    lists = sorted(diff.items())
    x,y = zip(*lists)
    plt.xlabel('k')
    plt.ylabel('value')
    plt.plot(x,y)
    plt.show()
