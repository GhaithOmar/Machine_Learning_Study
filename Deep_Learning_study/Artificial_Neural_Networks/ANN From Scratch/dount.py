import numpy as np
import matplotlib.pyplot as plt


def forward(X, W1, b1, W2, b2):
    # sigmoid
    #Z = 1 / (1+np.exp(-(X.dot(W1)+b1)))
    # Tanh
    Z = np.tanh(X.dot(W1) + b1)
    # relu
    #Z = X.dot(W1) + b1
    #Z = Z * (Z > 0)
    a = Z.dot(W2) + b2
    Y = 1 / (1 + np.exp(-a))
    return Y, Z


def predict(X, W1, b1, W2, b2):
    Y, _ = forward(X, W1, b1, W2, b2)
    return np.round(Y)


def deriv_W2(Z, T, Y):
    return Z.T.dot(T - Y)


def deriv_b2(T, Y):
    return (T - Y).sum()


def deriv_W1(X, Z, T, Y, W2):
    # sigmoid
    #dZ = np.outer(T-Y,W2) * Z*(1-Z)
    # tanh
    dZ = np.outer(T - Y, W2) * (1 - Z * Z)
    # dZ = np.outer(T-Y, W2) * (Z > 0) # this is for relu activation
    return X.T.dot(dZ)


def deriv_b1(Z, T, Y, W2):
    # tanh
    dZ = np.outer(T - Y, W2) * (1 - Z * Z)
    # sigmoid
    #dZ = np.outer(T-Y,W2) * Z*(1-Z)
    # relu
    # dZ = np.outer(T-Y, W2) * (Z > 0) # this is for relu activation

    return dZ.sum(axis=0)


def cost(T, Y):
    return np.sum(T * np.log(Y) + (1 - T) * np.log(1 - Y))


def test_xor():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = np.array([0, 1, 1, 0])
    W1 = np.random.randn(2, 5)
    b1 = np.random.randn(5)
    W2 = np.random.randn(5)
    b2 = np.random.randn(1)
    LL = []  # keep track of likelihoods

    learning_rate = 0.0005
    regularization = 0.
    last_error_rate = None

    for i in range(100000):
        pY, Z = forward(X, W1, b1, W2, b2)

        ll = cost(Y, pY)
        prediction = predict(X, W1, b1, W2, b2)

        er = np.abs(prediction - Y).mean()

        if er != last_error_rate:
            last_error_rate = er
            print(f"error rate is : {er}")
            print(f'true : {Y}')
            print(f'pred: {prediction}')
        if LL and ll < LL[-1]:
            print("early exit")
            break

        LL.append(ll)
        W2 += learning_rate * (deriv_W2(Z, Y, pY) - regularization * W2)
        b2 += learning_rate * (deriv_b2(Y, pY) - regularization * b2)
        W1 += learning_rate * (deriv_W1(X, Z, Y, pY, W2) - regularization * W1)
        b1 += learning_rate * (deriv_b1(Z, Y, pY, W2) - regularization * b1)

        if i % 10000 == 0:
            print(f'll is : {ll}')

    print(f"Final Classification rate: {(1-np.abs(prediction-Y).mean())}")
    plt.plot(LL)
    plt.show()


def test_donut():
    N = 1000
    R_inner = 5
    R_outer = 10

    # distance from origin is radius + random normal
    # angle theta is uniformly distributed between (0, 2pi)

    R1 = np.random.randn(N // 2) + R_inner
    theta = 2 * np.pi * np.random.random(N // 2)
    X_inner = np.concatenate([[R1 * np.cos(theta)], [R1 * np.sin(theta)]]).T

    R2 = np.random.randn(N // 2) + R_outer
    theta = 2 * np.pi * np.random.random(N // 2)
    X_outer = np.concatenate([[R2 * np.cos(theta)], [R2 * np.sin(theta)]]).T

    X = np.concatenate([X_inner, X_outer])
    Y = np.array([0] * (N // 2) + [1] * (N // 2))
    n_hidden = 10

    W1 = np.random.randn(2, n_hidden)
    b1 = np.random.randn(n_hidden)
    W2 = np.random.randn(n_hidden)
    b2 = np.random.randn(1)
    LL = []  # Keep track of likelihoods

    learning_rate = 0.00003
    regularization = 0.2
    for i in range(30000):
        pY, Z = forward(X, W1, b1, W2, b2)

        ll = cost(Y, pY)
        prediction = predict(X, W1, b1, W2, b2)
        er = np.abs(prediction - Y).mean()
        LL.append(ll)
        W2 += learning_rate * (deriv_W2(Z, Y, pY) - regularization * W2)
        b2 += learning_rate * (deriv_b2(Y, pY) - regularization * b2)
        W1 += learning_rate * (deriv_W1(X, Z, Y, pY, W2) - regularization * W1)
        b1 += learning_rate * (deriv_b1(Z, Y, pY, W2) - regularization * b1)

        if i % 100 == 0:
            print(f'll: {ll}', f" Classification rate : {1-er}")
    print(f"Final Classification rate: {(1-np.abs(prediction-Y).mean())}")
    plt.plot(LL)
    plt.show()


if __name__ == '__main__':
    test_xor()
    # test_donut()
