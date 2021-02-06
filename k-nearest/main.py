import numpy as np
from collections import Counter
import time
from functools import lru_cache

60, 120

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
    return data, tuple(labels)


def get_normalisation(data_set):
    """ Finds the normalisation of data_set and returns it as a List of highest values for each attribute
        and a list of the lowest value for each attribute.
    """
    max_values = np.array([np.max(column) for column in data_set.T])
    min_values = np.array([np.min(column) for column in data_set.T])
    return max_values, min_values


def normalize(min_values, max_values, data_set):
    """ Applies norm_range to the data_set to normalize it."""
    return tuple(tuple((feature - min_values) / (max_values - min_values)) for feature in data_set)


def get_distance(point_a, point_b):
    """ Gets distance between 2 points """
    distances         = np.array(point_a) - np.array(point_b)
    squared_distances = np.square(distances)
    summed_distance   = np.sum(squared_distances)
    return summed_distance


def k_nearest_neighbour(k, data_point, training_set, training_labels):
    """ Finds the k nearest neighbour points to data_point from training_set """
    # Get nearest labels to data_point
    classifications = get_nearest_labels(data_point, training_set, training_labels)

    # Return most common classification
    return Counter(classifications[:k]).most_common(1)[0][0]


@lru_cache(maxsize=256)
def get_nearest_labels(data_point, training_set, training_labels):
    """ Get labels of points nearest to data_point """
    # Calculate distance for each training_point
    distances = [get_distance(data_point, point) for point in training_set]

    # Sort training_labels based on distances
    return tuple(np.array(training_labels)[np.argsort(distances)])


def find_best_k(data_set, data_labels, validation_set, validation_labels):
    """find the best k in range of validation set
    Args:
        validation_set : list of points to validate
        validation_labels : labels for the validation set
        data_set : list of points to test against
        data_labels : labels for the test set
    Returns:
        int: k with best accuracy
    """
    best_k = 0
    best_value = 0

    for k in range(1, len(validation_set)):

        classifications = [k_nearest_neighbour(k, point, data_set, data_labels) for point in validation_set]
        result = np.mean(np.array(classifications) == np.array(validation_labels))

        print(f"k: {k},\taccuracy: {round(result * 100) }%")

        if result > best_value:
            best_value = result
            best_k = k

    print(f"best k is {best_k} with a {round(best_value * 100)}% accuracy")
    return best_k


if __name__ == '__main__':
    # load data from csv
    data_set, data_labels = load_data(r'dataset1.csv')
    validation_set, validation_labels = load_data(r'validation1.csv')
    days, _ = load_data(r'days.csv')

    # get normalisation  and apply to dataset
    min_values, max_values = get_normalisation(validation_set)
    data_set       = normalize(min_values, max_values, data_set)
    validation_set = normalize(min_values, max_values, validation_set)
    days           = normalize(min_values, max_values, days)

    # find best k
    best_k = find_best_k(data_set, data_labels, validation_set, validation_labels)

    # # Apply best_k to days that we do not have the date of and guess the season.
    season_guesses = [k_nearest_neighbour(best_k, day, validation_set, validation_labels) for day in days]
    print(season_guesses)

    print(f"Exectution took {time.perf_counter()} seconds")