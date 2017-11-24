import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

data = np.genfromtxt("yourfile2.csv", delimiter=',')

X = data[:,1:24]
y = data[:,24]

print(data[0,1:24])

# fit an Extra Trees model to the data
model = ExtraTreesClassifier()
model.fit(X, y)
# display the relative importance of each attribute
print(model.feature_importances_)

# # Load the Pima Indians diabetes dataset from CSV URL
# import numpy as np
# import urllib.request
# # URL for the Pima Indians Diabetes dataset (UCI Machine Learning Repository)
# url = "http://goo.gl/j0Rvxq"
# # download the file
# raw_data = urllib.request.urlopen(url)
# # load the CSV file as a numpy matrix
# dataset = np.loadtxt(raw_data, delimiter=",")
# print(dataset.shape)
# separate the data from the target attributes
# X = dataset[:,0:7]
# y = dataset[:,8]

# print(X)
# print(y)
