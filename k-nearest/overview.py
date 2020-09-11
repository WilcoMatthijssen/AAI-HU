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


def get_distance(point_a, point_b):
    """ Calculates distance between point a and b
          Args:
              point_a: list of features of a
              point_b: list of features of b
          Returns:
              float: distance between point a and b
    """


def get_normalisation(data_set):
    """ Finds the normalisation of data_set and returns it as a List of highest values for each attribute
        and a list of the lowest value for each attribute.
    """


def normalize(data_set, norm_range):
    """ Applies norm_range to the data_set to normalize it."""


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