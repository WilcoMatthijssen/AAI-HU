import numpy as np

dates = np.genfromtxt("days.csv", delimiter=";", usecols=[0])
labels = []
print(dates)
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


print(labels)