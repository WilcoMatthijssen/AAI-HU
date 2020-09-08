import numpy as np


def load_data(filename):
    data = np.genfromtxt(filename, delimiter=";", usecols=[1, 2, 3, 4, 5, 6, 7],
                         converters={5: lambda s: 0 if s == b"-1" else float(s),
                                     7: lambda s: 0 if s == b"-1" else float(s)})

    dates = np.genfromtxt(filename, delimiter=";", usecols=[0])
    labels = []
    for label in dates:
        if label < 20000301:
            labels.append("winter")
        elif 20000301 <= label < 20000601:
            labels.append("lente")
        elif 20000601 <= label < 20000901:
            labels.append("zomer")
        elif 20000901 <= label < 20001201:
            labels.append("herfst")
        else:  # from 01-12 to end of year
            labels.append("winter")
    return data, labels


def normalisation():
    data_set, data_labels = load_data("dataset1.csv")
    validation_set, validation_labels = load_data("validation1.csv")

    norm_range = get_normalisation(data_set)

    norm_data = normalize(data_set, norm_range)
    norm_validation_data = normalize(validation_set, norm_range)

    np.set_printoptions(threshold=np.inf)
    print(np.round(norm_data, 2))


def get_distance(a, b):
    # geen idee of dit zo hoort of nodig is. (wordt niet gebruikt)
    total = 0
    for i in range(7):
        total += (a[i]-b[i])**2
    return np.sqrt(total)


def get_normalisation(data):
    max_values = []
    min_values = []
    for column in data.T:
        max_values.append(np.max(column))
        min_values.append(np.min(column))
    return max_values, min_values


def normalize(data, norm_range):
    min_values, max_values = norm_range
    for i in range(len(min_values)):
        for feature in data:
            feature[i] = (feature[i] - min_values[i]) / (max_values[i] - min_values[i])
    return data


normalisation()
