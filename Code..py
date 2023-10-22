# Importing Libraries
import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Reading Data
df = pd.read_csv('C:\\Users\\dell\\Downloads\\Boston_Test.csv')

 # Data Exploration
df.shape
df.describe()
df.info()
df.isnull().sum().sum()
df['chas'].value_counts()
df['zn'].value_counts()
df.dtypes
df.corr()
plt.figure(figsize=(15, 8))
sns.heatmap(df.corr(), annot=True)
plt.show()

 # Data Visualization
y3 = df['medv']
for i in df.columns:
    x3 = df[i]
    plt.scatter(x3, y3)
    plt.show()

# Data Preprocessing
x_train = df.drop(['medv'], axis=1)
y_train = df['medv']

# Data Modeling (Linear Regression)
lr = LinearRegression()
lr.fit(x_train, y_train)
df2 = pd.read_csv("C:\\Users\\dell\\Downloads\\Boston_Test.csv")
x_test = df2.drop(['medv'], axis=1)
y_test = df2['medv']
y_pred = lr.predict(x_test)
plt.scatter(y_test, y_pred)
plt.show()
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
mse = mean_squared_error(y_test, y_pred)
ae = mean_absolute_error(y_test, y_pred)
re = r2_score(y_test, y_pred)
print(mse)
print(ae)
print(re)

 # Random Forest Regression
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=100, random_state=0)
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)
plt.scatter(y_test, y_pred)
plt.show()
regressor.score(x_test, y_test)
