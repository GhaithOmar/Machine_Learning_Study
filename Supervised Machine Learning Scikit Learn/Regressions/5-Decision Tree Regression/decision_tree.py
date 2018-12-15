import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv("Position_Salaries.csv")
X = df.iloc[:, 1:2].values
Y = df.iloc[:, -1].values


from sklearn.tree import DecisionTreeRegressor

reg = DecisionTreeRegressor(random_state=0)
reg.fit(X, Y)

yhat = reg.predict(np.array([[6.5]]))
print(yhat)


X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape(len(X_grid), 1)

plt.scatter(X, Y, color='red')
plt.plot(X_grid, reg.predict(X_grid), color='blue')
plt.title('Higher Resolution visulization')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
