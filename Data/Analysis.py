import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from datetime import date
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, neighbors

df = pd.read_csv("../Data/Confirmed.csv")
df = df.set_index(df['Date'])
#df.drop(['Date'], 1, inplace=True)
df.drop(['Status'], 1, inplace=True)
#print(df)
#plt.scatter(df.index, df[['MH']])
##plt.plot(df[['MH']])
##plt.legend()
##plt.show()

new_data = df[['MH']]
dataset = new_data.values

train = dataset
valid = dataset

#converting dataset into x_train and y_train
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

x_train, y_train = [], []
for i in range(10, len(train)):
    x_train.append(scaled_data[i-10:i,0])
    y_train.append(scaled_data[i,0])
x_train, y_train = np.array(x_train), np.array(y_train)

x_test, y_test = [], []
for i in range(10, len(valid)):
    x_test.append(scaled_data[i-10:i,0])
    y_test.append(scaled_data[i,0])
x_test, y_test = np.array(x_test), np.array(y_test)

from sklearn import linear_model

# Linear Regression
linear_regr = linear_model.LinearRegression()

# Train the model using the training sets
linear_regr.fit(x_train, y_train)
print(linear_regr.score(x_train, y_train))

lr_prediction = linear_regr.predict(x_train)
#print(lr_prediction)

plt.plot(lr_prediction)
plt.legend()
plt.show()
