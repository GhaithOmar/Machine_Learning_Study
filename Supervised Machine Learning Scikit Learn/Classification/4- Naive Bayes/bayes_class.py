import numpy as np
from scipy.stats import multivariate_normal as mvn


class Bayes(object):

    def fit(self, X, Y, smoothing=10e-3):
        N, D = X.shape
        self.gaussians = dict()
        self.priors = dict()
        labels = set(Y)
        for c in labels:
            Xc = X[Y == c]
            self.gaussians[c] = {
                'mean': Xc.mean(axis=0),
                'cov': np.cov(Xc.T) + np.eye(D) * smoothing
            }

            self.priors[c] = float(len(Y[Y == c])) / len(Y)

    def score(self, X, Y):
        P = self.predict(X)
        return np.mean(Y == P)

    def predict(self, X):
        N, D = X.shape
        k = len(self.gaussians)
        P = np.zeros((N, k))
        for c, g in self.gaussians.items():
            mean, cov = g['mean'], g['cov']
            P[:, c] = mvn.logpdf(X, mean=mean, cov=cov) + \
                np.log(self.priors[c])

        return np.argmax(P, axis=1)


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd

    # Importing the dataset
    dataset = pd.read_csv('Social_Network_Ads.csv')
    X = dataset.iloc[:, [2, 3]].values
    y = dataset.iloc[:, 4].values

    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0)

    # Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    classifier = Bayes()
    classifier.fit(X_train, y_train)
    ypred = classifier.predict(X_test)
    print(classifier.score(X_test, y_test))
