import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn. model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


# loading the csv file
data = pd.read_csv('OverallPlayerPerformancesBidsNewUpdated.csv')

playerNames = data['Player_Name']

# for dropping multiple columns
df = data.drop(['Pos', 'Player_Name', 'Team', 'Team_Symbol'], axis=1)

# extracting necessary data into features and the predicting values
Y = df['Bid_Values']
X = df.drop(['Bid_Values'], axis=1)

# seperating randomly selected 80% data as training data set and other 20% as the testing dataset
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2)
print(X_train)
print(X_val)
# X_train = X_train.drop(['Player_Name'], axis=1)
# X_val = X_val.drop(['Player_Name'], axis=1)
# Y_train = Y_train.drop(['Player_Name'], axis=1)
# Y_target = Y_val
# Y_val = Y_val.drop(['Player_Name'], axis=1)

######################################################
# # applying linear regression for modelin of the problem
# lin_reg = LinearRegression()
# lin_reg.fit(X_train, Y_train)
# prob = lin_reg.predict(X_val)  # predicting the values for the testing dataset
# mse = mean_squared_error(prob, Y_val)  # Calculating the mean square error with the test data and the predicted data

###########################################################
#applying polynomial regression for the model
poly = PolynomialFeatures(degree=2)
X_ = poly.fit_transform(X_train)
#X__ = poly.fit_transform(X)
X_test_ = poly.fit_transform(X_val)

# Instantiate
lg = LinearRegression()
# Fit
lg.fit(X_, Y_train)
# Predict

prob = lg.predict(X_test_)
#prob = lg.predict(X__)
# prob = lin_reg.predict(X_val)  # predicting the values for the testing dataset
mse = mean_squared_error(prob, Y_val)  # Calculating the mean square error with the test data and the predicted data
print(r2_score(Y_val, prob))

#print(act)
print(prob)
print(mse)

playerDetails = []
act = []

for num in range(0, len(Y_val)):
    act.append(Y_val[Y_val.index[num]])
    playerDetails.append(playerNames[Y_val.index[num]])

print(playerDetails)
print(act)


