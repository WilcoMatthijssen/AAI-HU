import numpy as np
from collections import Counter


def load_data(filename):
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




def get_distance(a, b):
    # geen idee of dit zo hoort of nodig is. (wordt niet gebruikt)
    total = 0
    for i in range(7):
        total += (a[i] - b[i])**2
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
            feature[i] = (feature[i] - min_values[i]) / (
                max_values[i] - min_values[i])
    return data


def k_nearest_neighbour(k, data_point, training_set, training_labels):
    distances = []
    for training_point, training_label in zip(training_set, training_labels):
        distances.append( [get_distance(data_point, training_point), training_label] )

    # Sort based on distance
    distances.sort()

    # Get K lowest distances classifications
    classifications =[ i[1] for i in distances[:k] ]

    # Return most common classification
    return Counter(classifications).most_common(1)[0][0]



# def find_best_k(validation_set, validation_labels, data_set, data_labels):
    # voor elke 1 tot len val_set (K)
    #     test elk element val_set en check of classification klopt
    #     tell dan 1 bij match
    #     kijk of hoogste match is bereikt
    #     gebruik daar de beste van



#     best_k = 0
#     highest_value = 0.0

#     for k in range(1, len(validation_set)):

#     for k in range(1, len(v_set)):
#         classifications = []
#         for validation_point in v_set:
#             classifications.append(k_nearest_neighbour(k, validation_point,
#                                                        t_set, t_labels))
#         print("done11111")
#         matches: int = 0
#         for classification_index, _ in enumerate(classifications):
#             if classifications[classification_index] == v_labels[classification_index]:
#                 matches += 1
#         res: float = matches * 100 / len(v_labels)
        

#         if res > highest_value:
#             highest_value = res
#             best_k = k
    
#     return highest_k



if __name__ == '__main__':
    print("start")
    data_set, data_labels = load_data("dataset1.csv")
    validation_set, validation_labels = load_data("validation1.csv")

    normalize_range = get_normalisation(data_set)

    normalized_data = normalize(data_set, normalize_range)
    normalized_val_data = normalize(validation_set, normalize_range)

    #best_k = find_best_k(normalized_data, data_labels, normalized_val_data, validation_labels)
    print(k_nearest_neighbour(3,normalized_val_data[0], normalized_data, data_labels))

    np.set_printoptions(threshold=np.inf)
    #print(np.round(normalized_data, 2))

   
    print("done")

    #print(data_labels)


