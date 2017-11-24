#!/usr/bin/pytho

import pandas as pand
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score

def load_data():
   df = pand.read_csv("yourfile1.csv");
   return df

df = load_data()
df.drop(['dataId'],inplace=True, axis=1)

column_names = df.columns.values
print(len(column_names))
index = [len(column_names)-1]
#index = [1,2,3,4,5,6,11]
features = np.delete(column_names, index)

# separating 80% data for training
train = df.sample(frac=0.8, random_state=1)
#print(train)

# rest 20% data for evaluation purpose
test = df.loc[~df.index.isin(train.index)]
#print(test)

#using the seperated 80% train data set devide the features and lables
features_train = train[features]
labels_train = train["result"]
features_test = test[features]
labels_test = test["result"]

# runRate_lose       = [features_train[0] for ii in range(0, len(features_train)) if labels_train==0]
# powerPlayRuns_lose = [features_train[1] for ii in range(0, len(features_train)) if labels_train==0]
# runRate_win        = [features_train[0] for ii in range(0, len(features_train)) if labels_train==1]
# powerPlayRuns_win  = [features_train[1] for ii in range(0, len(features_train)) if labels_train==1]
#
# #### initial visualization
# plt.xlim(0.0, 90.0)
# plt.ylim(0.0, 15.0)
# plt.scatter(powerPlayRuns_lose, runRate_lose, color = "b", label="lose")
# plt.scatter(powerPlayRuns_win, runRate_win, color = "r", label="win")
# plt.legend()
# plt.xlabel("dots")
# plt.ylabel("runs")
# plt.show()
# print("asdfasdf")

import itertools as iter

def pset(lst):
    comb = (iter.combinations(lst, l) for l in range(len(lst) + 1))
    return list(iter.chain.from_iterable(comb))

newArray = pset(features)
from sklearn.svm import SVC
clf = SVC(kernel="rbf", C=10000, gamma=1)
print("begin feature len", len(newArray))
myList=[]
featureArray = []

#for num in range(0, len(newArray)-1):  # Second Example

newArray = pset(features)#creating all possible combinations to array
for num in range(0, 1000):
#for num in range(0, len(newArray)):
    print('Combination of :', newArray[num+1], ' number ',  num)
    featureArray.append(newArray[num+1])
    features_train = train[np.asarray(newArray[num+1])]#selecting a combination of features
    labels_train = train["result"]
    features_test = test[np.asarray(newArray[num + 1])]
    labels_test = test["result"]
    clf.fit(features_train, labels_train)#fitting the data to learner
    predictions = clf.predict(features_test)#predicting for test data
    mse = accuracy_score(predictions, labels_test)
    # mse = mean_squared_error(predictions, labels_test)
    myList.append(mse)


    print("heee your done")
    print(mse)

print(myList)


mylist1 = list(range(len(myList)))
print(mylist1)

#### initial visualization
print(featureArray)
print(features)
#plt.xticks(x, my_xticks)
plt.xlim(0.0, len(myList))
plt.ylim(0.0, 1.0)
#plt.scatter(mylist1, myList, color = "b", label="lose")
plt.plot(mylist1, myList, 'b-',label='Accuracy deviance')
#plt.scatter(powerPlayRuns_win, runRate_win, color = "r", label="win")
plt.legend()
plt.xlabel("possible combinations (1000)")
plt.ylabel("accuracy")
plt.show()
print("asdfasdf")


x = mylist1
y = myList
#x_ticks_labels = ['jan','feb','mar','apr','may']
x_ticks_labels = features

fig, ax = plt.subplots()
fig.subplots_adjust(bottom=0.25)
ax.plot(x,y)

# Set number of ticks for x-axis
ax.set_xticks(x)
# Set ticks labels for x-axis
ax.set_xticklabels(x_ticks_labels, rotation='vertical', fontsize=10)
# fig.suptitle('Deviation of Mean Square Error', fontsize=14)
fig.suptitle('Deviation of accuracy', fontsize=14)
plt.xlabel("single combinations")
#plt.ylabel("mean square error")
plt.ylabel("accuracy")
plt.show()
print("doneeee")