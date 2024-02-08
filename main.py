import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

pd.set_option('display.max_columns', 200)

df = pd.read_csv('C:/Users/dtgiv/Downloads/team.csv')
print(df)

print(df.columns)

#do stuff here to investigate table grain

#filter data on 162-game seasons only
df = df.query('g == 162')
print(df)

#check for nulls
print(df.isnull().sum())

#define x
x = df['r']
print(x)

#define y
y = df['w']
print(y)

#doublecheck x for nulls
print(x.isnull().sum())

#doublecheck dimensions of x
print(len(x))

#doublecheck y for nulls
print(y.isnull().sum())

#doublecheck dimensions of y
print(len(y))

#visualize data
plt.scatter(x, y)
plt.title('Wins Per Runs')
plt.xlabel('Runs Per Team Per Season')
plt.ylabel('Wins Per Team Per Season')
plt.show()

#split x and y into training and test arrays
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.4, random_state = 23)

print(x_train)
print(x_test)
print(y_train)
print(y_test)

x_train = np.array(x_train).reshape(-1, 1)
print(x_train)

x_test = np.array(x_test).reshape(-1, 1)
print(x_test)

#save LinearRegression as variable lr
lr = LinearRegression()

#train the model
lr.fit(x_train, y_train)

#what's the y-intercept
c = lr.intercept_
print(c)

#what's the slope
m = lr.coef_
print(m)

#test the model
y_pred_train = m*x_train + c
print(y_pred_train)
