#!/usr/bin/pytho

import pandas as pand
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error

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
leastArray = []
leastArrayNew = []
leastArrayNames = []
leastArrayNamesNew = []

#for num in range(0, len(newArray)-1):  # Second Example

newArray = pset(features)#creating all possible combinations to array
print(len(newArray))

def checkForLowestTen(lst,lstNames, mse, combination):
    print("lis" , lst)
    print("combinations", lstNames)
    if len(lst) == 10:
        for num in range(0, len(lst)):
            if(mse <= lst[num]):
                if num > 0:
                    lst[num-1] = lst[num]
                    lstNames[num-1] = lstNames[num]
                lst[num] = mse
                lstNames[num] = combination

            else:
                break
        print("hellooo broo in the all completed list")
    elif len(lst) > 0:
        for num in range(0, len(lst)):
            if(mse <= lst[num]):
                if num == len(lst)-1:
                    lst.append(mse)
                    lstNames.append(combination)
            else:
                lst.insert(num, mse)
                lstNames.insert(num, combination)
                break
        print("hii bro your not completed the whole list")
    elif len(lst) == 0:
        lst.append(mse)
        lstNames.append(combination)
        print("you have null value")

    return lst, lstNames

#for num in range(0, 15):
for num in range(2377000, 2378000):
    print('Combination of :', newArray[num+1], ' number ',  num+1)
    featureArray.append(newArray[num+1])
    features_train = train[np.asarray(newArray[num+1])]#selecting a combination of features
    labels_train = train["result"]
    features_test = test[np.asarray(newArray[num + 1])]
    labels_test = test["result"]
    clf.fit(features_train, labels_train)#fitting the data to learner
    predictions = clf.predict(features_test)#predicting for test data
    mse = mean_squared_error(predictions, labels_test)
    myList.append(mse)

    leastArray, leastArrayNames = checkForLowestTen(leastArray, leastArrayNames, mse, newArray[num+1])

    print("heee your done")
    print(mse)

#print(myList)
eee = sorted(myList)
#print(eee)
#print(eee[2])

mylist1 = list(range(len(myList)))
mylist2 = list(range(len(leastArray)))
#print(mylist1)

#### initial visualization
#plt.xticks(x, my_xticks)
plt.xlim(0.0, len(mylist2))
plt.ylim(0.0, 1.0)
#plt.scatter(mylist1, myList, color = "b", label="lose")
plt.plot(mylist2, leastArray, '--bo', label='Lowest Mean Square Error deviance')
#plt.scatter(mylist2, leastArray,label='Mean Square Error deviance')
#plt.scatter(powerPlayRuns_win, runRate_win, color = "r", label="win")
plt.legend()
plt.xlabel("lowest possible combinations (20)")
plt.ylabel("mean square error")
plt.show()
print("asdfasdf")
#
#

#plt.xticks(x, my_xticks)
plt.xlim(0.0, len(myList))
plt.ylim(0.0, 1.0)
#plt.scatter(mylist1, myList, color = "b", label="lose")
plt.plot(mylist1, myList, '--bo',label='Mean Square Error deviance')
#plt.scatter(mylist2, leastArray,label='Mean Square Error deviance')
#plt.scatter(powerPlayRuns_win, runRate_win, color = "r", label="win")
plt.legend()
plt.xlabel("lowest possible combinations (20)")
plt.ylabel("mean square error")
plt.show()
print("asdfasdf")
# x = mylist1
# y = myList
# #x_ticks_labels = ['jan','feb','mar','apr','may']
# x_ticks_labels = features
#
# fig, ax = plt.subplots()
# fig.subplots_adjust(bottom=0.25)
# ax.plot(x,y)
#
# # Set number of ticks for x-axis
# ax.set_xticks(x)
# # Set ticks labels for x-axis
# ax.set_xticklabels(x_ticks_labels, rotation='vertical', fontsize=10)
# fig.suptitle('Deviation of Mean Square Error', fontsize=14)
# plt.xlabel("single combinations")
# plt.ylabel("mean square error")
# plt.show()
# print("doneeee")