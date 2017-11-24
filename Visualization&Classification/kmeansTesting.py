#!/usr/bin/pytho

import pandas as pand
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error

def load_data():
   df = pand.read_csv("kmeans.csv");
   return df

df = load_data()
df.drop(['nu'],inplace=True, axis=1)
df.drop(['playerId'],inplace=True, axis=1)

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
features_test = test[features]

