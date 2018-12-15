import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from process import get_data


def y2indicator(y, K):
    N = len(y)
    ind = np.zeros((N, K))
    for i in range(N):
        ind[i, y[i]] = 1
    return ind


def forward(X, W1, b1, W2, b2):
    Z = 1 / (1 + np.exp(-X.dot(W1) - b1))

    expA = np.exp(Z.dot(W2) + b2)
    Y = expA / expA.sum(axis=1, keepdims=True)
    return Y, Z


def dirv_W2(Z, T, Y):
    return Z.T.dot(T - Y)


def dirv_b2(T, Y):
    return (T - Y).sum(axis=0)


def dirv_W1(X, Z, T, Y, W2):
    dZ = (T - Y).dot(W2.T) * Z * (1 - Z)
    return X.T.dot(dZ)


def dirv_b1(Z, T, Y, W2):
    return ((T - Y).dot(W2.T) * Z * (1 - Z)).sum(axis=0)


def cost(T, Y):
    return -np.mean(T * np.log(Y))


def score(Y, P):
    n_correct = 0
    n_total = 0

    for i in range(len(Y)):
        n_total += 1
        if Y[i] == P[i]:
            n_correct += 1
    return float(n_correct) / n_total


def main():
    X, Y = get_data()
    X, Y = shuffle(X, Y)
    Y = Y.astype(np.int32)  # conver y to int32

    D = X.shape[1]
    K = len(set(Y))
    M = 5
    Xtrain = X[:-100]
    Ytrain = Y[:-100]
    T_train = y2indicator(Ytrain, K)
    Xtest = X[-100:]
    Ytest = Y[-100:]
    T_test = y2indicator(Ytest, K)

    W1 = np.random.randn(D, M)
    b1 = np.random.randn(M)
    W2 = np.random.randn(M, K)
    b2 = np.random.randn(K)

    learning_rate = 0.001
    train_costs = []
    for epoch in range(10000):
        output, hidden = forward(Xtrain, W1, b1, W2, b2)
        if epoch % 100 == 0:
            c_train = cost(T_train, output)
            P = np.argmax(output, axis=1)
            r = score(Ytrain, P)
            print(
                f'epoch #: {epoch}     Train Cost: {c_train}      score: {r}')
            train_costs.append(c_train)

        W2 += learning_rate * dirv_W2(hidden, T_train, output)
        b2 += learning_rate * dirv_b2(T_train, output)
        W1 += learning_rate * dirv_W1(Xtrain, hidden, T_train, output, W2)
        b1 += learning_rate * dirv_b1(hidden, T_train, output, W2)

    plt.plot(train_costs)
    plt.show()
    output, _ = forward(Xtest, W1, b1, W2, b2)
    yhat = np.argmax(output, axis=1)
    print(f'Test Score: {score(Ytest,yhat)}')


if __name__ == '__main__':
    main()
