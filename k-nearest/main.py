import numpy as np
from collections import Counter
import time


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
        if label < '0301':
            labels.append("winter")
        elif '0301' <= label < '0601':
            labels.append("lente")
        elif '0601' <= label < '0901':
            labels.append("zomer")
        elif '0901' <= label < '1201':
            labels.append("herfst")
        else:  # from 01-12 to end of year
            labels.append("winter")
    return data, labels


def get_distance(point_a, point_b):
    """ Calculates distance between point a and b
          Args:
              point_a: list of features of a
              point_b: list of features of b
          Returns:
              float: distance between point a and b
    """
    distance = 0.0
    for ax, bx in zip(point_a, point_b):
        distance += (ax - bx)**2
    return distance


def get_normalisation(data_set):
    """ Loads a csv file for use by k_nn
          Args:
              data_set: list of points
          Returns:
              tuple[list,list]: List of highest values in each column and a list of the lowest value in each column.
    """
    max_values = []
    min_values = []
    for column in data_set.T:
        max_values.append(np.max(column))
        min_values.append(np.min(column))
    return max_values, min_values

def normalize(data_set, norm_range):
    """ Loads a csv file for use by k_nn
          Args:
              data_set: data to normalize
              norm_range: tuple containing highest and lowest value for each column of data
          Returns:
              list: A list of normalized data_set
          """
    min_values, max_values = norm_range
    for i in range(len(max_values)):
        for feature in data_set:
            feature[i] = (feature[i] - min_values[i]) / (max_values[i] - min_values[i])
    return data_set


def k_nearest_neighbour(k, data_point, training_set, training_labels):
    """Finds the k nearest neighbour points to data_point from training_set
    Args:
        k: nearest neighbour count
        data_point: a list of features to classify
        training_set: the training set from which to find the nearest points
        training_labels: list of labels to assign to t_point from
    Returns:
        str: A probable correct classification for data_point
    """
    distances = []
    for training_point, training_label in zip(training_set, training_labels):
        distances.append( [get_distance(data_point, training_point), training_label] )

    # Sort based on distance
    distances.sort()

    # Get K lowest distances classifications
    classifications = [ i[1] for i in distances[:k] ]

    # Return most common classification
    return Counter(classifications).most_common(1)[0][0]


def find_best_k(validation_set, validation_labels, data_set, data_labels):
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
    for k in range(1, len(validation_set), 1 ):
        classifications = []
        for validation_point in validation_set:
            classifications.append(k_nearest_neighbour(k, validation_point,
                                                       data_set, data_labels))
        matches = 0
        for classification_index, _ in enumerate(classifications):
            if classifications[classification_index] == validation_labels[classification_index]:
                matches += 1
        result = (matches * 100) / len(validation_labels)
        print(k,result)
        if result > best_value:
            best_value = result
            best_k = k
    print("best k is {} with a {}% accuracy".format(best_k,best_value))
    return best_k

if __name__ == '__main__':
    #Had to use r'' because \v is a character.
    data_set, data_labels = load_data(r'dataset1.csv')
    validation_set, validation_labels = load_data(r'validation1.csv')
    days, _ = load_data(r'days.csv')

    normalize_range = get_normalisation(data_set)

    # normalize sets
    normalize(data_set, normalize_range)
    normalize(validation_set, normalize_range)
    normalize(days, normalize_range)

    best_k = find_best_k(data_set, data_labels, validation_set, validation_labels)
    #print(k_nearest_neighbour(3,normalized_val_data[0], normalized_data, data_labels))

    # Apply best_k to days that we do not have the date of and guess the season.
    season_guesses = [k_nearest_neighbour(best_k, day, validation_set, validation_labels) for day in days]
    print(season_guesses)



