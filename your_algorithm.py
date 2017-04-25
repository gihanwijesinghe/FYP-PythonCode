#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()
print(features_train)

### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()
################################################################################

### your code here!  name your classifier object clf if you want the
#######################     Naive Base  ##########################################################################################
# from sklearn.naive_bayes import GaussianNB
# clf = GaussianNB()
# clf.fit(features_train, labels_train)
# pred = clf.predict(features_test)
# pred2 = clf.predict([[0.2, 0.2]])
# pred3 = clf.predict([[0.4, 0.4]])
#
# print(pred2)
# print(pred3)
#################################################################################################################################3

########################    SVM             #######################################################################################
# from sklearn.svm import SVC
# clf = SVC(kernel="rbf", C=10000, gamma=1000)
# clf.fit(features_train, labels_train)
###################################################################################################################################





# from sklearn import datasets
# iris = datasets.load_iris()
# from sklearn.naive_bayes import GaussianNB
# gnb = GaussianNB()
# y_pred = gnb.fit(iris.data, iris.target).predict(iris.data)
# print("Number of mislabeled points out of a total %d points : %d" % (iris.data.shape[0],(iris.target != y_pred).sum()))





########################    Desicion Tree   ######################################################################################
from sklearn import tree
gnb = tree.DecisionTreeClassifier(min_samples_split=10)
gnb.fit(features_train, labels_train)
prettyPicture(gnb, features_test, labels_test)

plt.show()
###################################################################################################################################

### visualization code (prettyPicture) to show you the decision boundary
#
# from sklearn.ensemble import AdaBoostClassifier
# clf = AdaBoostClassifier(n_estimators=10, learning_rate=1)
# clf.fit(features_train, labels_train)
#
# from sklearn.ensemble import RandomForestClassifier
# clf = RandomForestClassifier(n_estimators=50)
# clf.fit(features_train, labels_train)

# from sklearn.neighbors import KNeighborsClassifier
#
# for x in range(1, 10):
#     clf = KNeighborsClassifier(n_neighbors=x)
#     clf.fit(features_train, labels_train)
#     accuracy = clf.score(features_test, labels_test)
#     print(x, accuracy)

#scores = cross_val_score(clf, iris.data, iris.target)
#print scores.mean()    






try:
    prettyPicture(clf, features_test, labels_test)
    print("yes got");
except NameError:
    pass
