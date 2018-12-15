import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

df = pd.read_csv('..\\Data\\housing_data.csv', header=None, sep=r'\s+')

X = df.iloc[:, :-1].values
Y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.3, random_state=0)

x_sc = StandardScaler()
y_sc = StandardScaler()
X_train = x_sc.fit_transform(X_train)
X_test = x_sc.transform(X_test)
y_test = np.log(y_test).reshape(-1, 1)
y_train = np.log(y_train).reshape(-1, 1)
y_train = y_sc.fit_transform(y_train).flatten()
y_test = y_sc.transform(y_test).flatten()


model = RandomForestRegressor(100)

model.fit(X_train, y_train)
yhat = model.predict(X_test)

plt.scatter(y_test, yhat)
plt.xlabel("target")
plt.ylabel("prediction")
ymin = np.round(min(min(y_test), min(yhat)))
ymax = np.ceil(max(max(y_test), max(yhat)))
r = range(int(ymin), int(ymax) + 1)
plt.plot(r, r)
plt.show()
plt.plot(y_test, label='targets')
plt.plot(yhat, label='prediction')
plt.legend()
plt.show()

print(model.score(X_test, y_test))

print(
    f'this cross_val_score for train: {cross_val_score(model,X_train,y_train).mean()}')
