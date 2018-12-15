import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Logistic_regression_class import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv('..\\Data\\Social_Network_Ads.csv')

X = df.iloc[:, 2:3].values
X_multi = df.iloc[:, 2:4].values
Y = df.iloc[:, -1].values

X_train, X_test, Y_train, Y_test = train_test_split(
    X_multi, Y, test_size=.28, random_state=0)

classifier = LogisticRegression()
classifier.fit(X_train, Y_train, times=5000, lr=0.002)

yhat = classifier.predict(X_test)
print("this is what i predict:\n", yhat)
print("this is what y actualy:\n", Y_test)
costs = classifier.costs
plt.plot(costs)
plt.show()
# plt.scatter(X_test[:,0],X_test[:,1])
# plt.show()
print(f'this is my classification rate {classifier.score(X_test,Y_test)}')
