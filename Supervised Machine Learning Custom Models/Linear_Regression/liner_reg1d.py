import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('..\\Data\\Salary_Data.csv')

X = df['YearsExperience'].values
y = df['Salary'].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 3)


def fit(X_train, y_train):
    dim = X_train.dot(X_train) - (X_train.mean() * X_train.sum())
    a = (y_train.dot(X_train) - y_train.mean() * X_train.sum()) / dim
    b = (y_train.mean() * X_train.dot(X_train) -
         X_train.mean() * y_train.dot(X_train)) / dim
    return a, b


a, b = fit(X_train, y_train)

y_hat = b + a * X_test
y_hat1 = b + a * X_train
d1 = y_test - y_hat
d2 = y_test - y_test.mean()

SSres = d1.dot(d1)
SStot = d2.dot(d2)

r2 = 1 - SSres / SStot

plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, y_hat1, color='blue')
plt.show()

print(r2)
