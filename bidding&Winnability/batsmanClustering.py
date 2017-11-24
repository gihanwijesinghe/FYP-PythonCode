import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
from sklearn.cluster import KMeans

X = np.array([[1, 2, 5],
              [5, 8, 5],
              [1.5, 1.8, 9],
              [8, 8, 8],
              [1, 0.6, 3],
              [9, 11, 9]])

# data = np.genfromtxt("OverallBatsmen1.csv", delimiter=',')
data = np.genfromtxt("OverallBatsmenShuffeled.csv", delimiter=',')
batsman_name = np.loadtxt("batsmanname1.csv", dtype=str)

# print(X)
# print(data)

kmeans = KMeans(n_clusters = 3)
kmeans.fit(data)

centroids = kmeans.cluster_centers_
labels = kmeans.labels_

print(centroids)
print(labels)

# goodbatsman1 = []
goodbatsman2 = []
goodbatsman3 = []
badbatsman = []


for num in range(0, len(data)):
    # if labels[num] == 3:
    #     goodbatsman1.append(batsman_name[num])
    if labels[num] == 1:
        goodbatsman2.append(batsman_name[num])
    elif labels[num] == 2:
        goodbatsman3.append(batsman_name[num])
    else:
        badbatsman.append(batsman_name[num])

#print(batsman_name)
# print(goodbatsman1)
print(goodbatsman2)
print(goodbatsman3)
print(badbatsman)