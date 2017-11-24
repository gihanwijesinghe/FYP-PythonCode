import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn. model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# loading the csv file
data = pd.read_csv('Overall_batsmen_with_bids.csv')

# for dropping multiple columns
df = data.drop(['Pos', 'Player_Name', 'Team', 'Team_Symbol'], axis=1)

# extracting necessary data into features and the predicting values
Y = df['Bid_Values']
X = df.drop(['Bid_Values'], axis=1)

# seperating randomly selected 80% data as training data set and other 20% as the testing dataset
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2)

######################################################3
# applying linear regression for modelin of the problem
lin_reg = LinearRegression()
lin_reg.fit(X_train, Y_train)
prob = lin_reg.predict(X_val)  # predicting the values for the testing dataset
mse = mean_squared_error(prob, Y_val)  # Calculating the mean square error with the test data and the predicted data

print(Y_val)
print(prob)
print(mse)


