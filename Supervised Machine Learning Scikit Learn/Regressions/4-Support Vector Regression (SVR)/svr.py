import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("Position_Salaries.csv")
X = df.iloc[:, 1:2].values
Y = df.iloc[:, -1].values

#############################################################################
#				Feature scalling
###########################################################################

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
Y = Y.reshape(-1, 1)
Y = sc_y.fit_transform(Y)
Y = Y.ravel()
from sklearn.svm import SVR

regressor = SVR(kernel='rbf')
regressor.fit(X, Y)

y_hat = sc_y.inverse_transform(
    regressor.predict(sc_X.transform(np.array([[6.5]]))))
print(y_hat)

plt.scatter(X, Y)
plt.plot(X, regressor.predict(X))
plt.show()
