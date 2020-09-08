import numpy as np
import numpy as np
#datasetfile = "dataset1.csv"
datasetfile = "validation1.csv"


data = np.genfromtxt(datasetfile, delimiter=";", usecols=[1,2,3,4,5,6,7], converters={5: lambda s: 0 if s == b"-1" else float(s), 7: lambda s: 0 if s == b"-1" else float(s)})


datasetfile = "dataset1.csv"
validationfile = "validation1.csv"

<<<<<<< HEAD
def loadData():
    data = np.genfromtxt(datasetfile, delimiter=";", usecols=[1,2,3,4,5,6,7], converters={5: lambda s: 0 if s == b"-1" else float(s), 7: lambda s: 0 if s == b"-1" else float(s)})
    dates = np.genfromtxt(datasetfile, delimiter=";", usecols=[0])
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
        else: # from 01-12 to end of year
            labels.append("winter")
    return data

def normalisatie(input_value = None):
    data = loadData()

    max_values = []
    min_values = []
    
    for i in range(len(data[0])):
        max = 0
        min = float("inf")
        for j in data:
            value = j[i]
            max = value if value > max else max
            min = value if value < min else min
        max_values.append(max)
        min_values.append(min)
    for line in applyNormalisatie(min_values,max_values):
        for value in line:
            print(value, end='\t')
        print()
		

def applyNormalisatie(min_values, max_values, data = loadData()):
    for i in range(len(data[0])):
        max = max_values[i]
        min = min_values[i]
        for j in data:
            j[i] = (j[i] - min) / (max - min)
    #print(repr(data))
    return data
            
normalisatie()
=======

dates = np.genfromtxt(datasetfile, delimiter=";", usecols=[0])
labels = []
for label in dates:
  print(type(label))
  if label < 20000301:
    labels.append("winter")
  elif 20000301 <= label < 20000601:
    labels.append("lente")
  elif 20000601 <= label < 20000901:
    labels.append("zomer")
  elif 20000901 <= label < 20001201:
    labels.append("herfst")
  else: # from 01-12 to end of year
    labels.append("winter")
print( labels)


>>>>>>> 63617cd6cbc889ed118b5008be2042424cfc6a2e
