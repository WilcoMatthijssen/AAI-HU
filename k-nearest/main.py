import numpy as np
import numpy as np
#datasetfile = "dataset1.csv"
datasetfile = "validation1.csv"


data = np.genfromtxt(datasetfile, delimiter=";", usecols=[1,2,3,4,5,6,7], converters={5: lambda s: 0 if s == b"-1" else float(s), 7: lambda s: 0 if s == b"-1" else float(s)})




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


