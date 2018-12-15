import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LinearRegression
df = pd.read_csv('50_Startups.csv')

X = df.iloc[:, :-1].values
Y = df.iloc[:, -1].values

label = LabelEncoder()
X[:, -1] = label.fit_transform(X[:, -1])
one_hot = OneHotEncoder(categorical_features=[-1])

X = one_hot.fit_transform(X).toarray()
# Avoid the dummy variable trap
X = X[:, 1:]
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=0)

lr = LinearRegression()

lr.fit(X_train, Y_train)
yhat = lr.predict(X_test)

# Building The optimal model using Backward elimination
# Manual way
# import statsmodels.formula.api as sm

##X = np.append(arr=X, values = np.ones(50).astype(int), axis=1)
X = np.append(arr=np.ones((50, 1)).astype(int), values=X, axis=1)

# X_opt = X[:,[0,1,2,3,4,5]]
# reg_ols = sm.OLS(endog = Y, exog = X_opt).fit()
# #print(reg_ols.summary())

# X_opt = X[:,[0,1,3,4,5]]
# reg_ols = sm.OLS(endog = Y, exog = X_opt).fit()
# #print(reg_ols.summary())

# X_opt = X[:,[0,3,4,5]]
# reg_ols = sm.OLS(endog = Y, exog = X_opt).fit()
# #print(reg_ols.summary())
# X_opt = X[:,[0,3,5]]
# reg_ols = sm.OLS(endog = Y, exog = X_opt).fit()
# #print(reg_ols.summary())


#####################################################################

# Automatic way

import statsmodels.formula.api as sm


def backwardElimination(x, y, SL):
    numVars = len(x[0])
    temp = np.zeros((50, 6)).astype(int)
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        adjR_before = regressor_OLS.rsquared_adj.astype(float)
        if maxVar > SL:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    temp[:, j] = x[:, j]
                    x = np.delete(x, j, 1)
                    tmp_regressor = sm.OLS(y, x).fit()
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)
                    if (adjR_before >= adjR_after):
                        x_rollback = np.hstack((x, temp[:, [0, j]]))
                        x_rollback = np.delete(x_rollback, j, 1)
                        print(regressor_OLS.summary())
                        return x_rollback
                    else:
                        continue
    regressor_OLS.summary()
    return x


SL = 0.05

X_opt = X[:, [0, 1, 2, 3, 4]]
X_Modeled = backwardElimination(X_opt, Y, SL)

print(X_Modeled[:5])
#######################################################


X_train, X_test, Y_train, Y_test = train_test_split(
    X_Modeled, Y, test_size=0.2, random_state=0)

lr = LinearRegression()

lr.fit(X_train, Y_train)
yhat = lr.predict(X_test)

# print(yhat)
# print(Y_test)
print(f'R_square = {lr.score(X_test,Y_test)}')
