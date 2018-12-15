import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def forward(X):
    Z = X.dot(W1) + b
    # relu
    Z = Z * (Z > 0)
    # tanh
    #Z = np.tanh(X.dot(W1))
    Y = Z.dot(W2) + c
    return Y, Z


def derv_W2(Z, T, Y):
    return (T - Y).dot(Z)


def derv_c(Y, Yhat):
    return (Y - Yhat).sum()


def derv_W1(X, Z, T, Y, W2):
    dZ = np.outer(T - Y, W2) * (Z > 0)
    return X.T.dot(dZ)


def derv_b(Z, Y, Yhat, W2):
    dZ = np.outer(Y - Yhat, W2) * (Z > 0)
    return dZ.sum(axis=0)


def cost(T, Y):
    return ((T - Y)**2).mean()


def score(Y, yhat):
    d1 = Y - yhat
    d2 = Y - Y.mean()
    SSres = d1.dot(d1)
    SStot = d2.dot(d2)
    return 1 - (SSres / SStot)


N = 500
X = np.random.random((N, 2)) * 4 - 2
Y = X[:, 0] * X[:, 1]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], Y)
plt.show()
D = X.shape[1]
M = 100
eta = 0.00001
W1 = np.random.randn(D, M) / np.sqrt(D)
b = np.zeros(M)
W2 = np.random.randn(M) / np.sqrt(M)
c = 0
costs = []
for epoch in range(200):
    yhat, hidden = forward(X)

    W2 += eta * derv_W2(hidden, Y, yhat)
    c += eta * derv_c(Y, yhat)
    W1 += eta * derv_W1(X, hidden, Y, yhat, W2)
    b += eta * derv_b(hidden, Y, yhat, W2)
    ct = cost(Y, yhat)
    r = cost(Y, yhat)
    costs.append(ct)

    if epoch % 25 == 0:
        print(f'cost is: {ct}  ', f'Score is {r}')

plt.plot(costs)
plt.title('Costs')
plt.show()

yhat, _ = forward(X)

print(f'final score is : {score(Y,yhat)}')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], Y)

# surface plot
line = np.linspace(-2, 2, 20)
xx, yy = np.meshgrid(line, line)
Xgrid = np.vstack((xx.flatten(), yy.flatten())).T
Yhat, _ = forward(Xgrid)
ax.plot_trisurf(Xgrid[:, 0], Xgrid[:, 1], Yhat,
                linewidth=0.2, antialiased=True)
plt.show()

Ygrid = Xgrid[:, 0] * Xgrid[:, 1]
R = np.abs(Ygrid - Yhat)

plt.scatter(Xgrid[:, 0], Xgrid[:, 1], c=R)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(Xgrid[:, 0], Xgrid[:, 1], R, linewidth=0.2, antialiased=True)
plt.show()
