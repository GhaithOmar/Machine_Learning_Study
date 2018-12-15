import numpy as np
from util import get_data, get_xor, get_donut
from datetime import datetime
from scipy.stats import norm
from scipy.stats import multivariate_normal as mvn


class NaiveBayes(object):
    """ Naive Bayes algorithm is grounded in probability,this algorithm uses Bayes rule and it consider all the input variables as independent.
            Bayes rule:
            P(y|x) = P(x|y)*p(y) / P(x)

            Multivariate Gaussian:

            P(X) = (1/sqrt(2*pi)^D * magintued of sigma) * exp(-*0.5 * (X-mu).T * (sigma^-1) * (X-mu) )
            X = vector input
            mu = vector mean
            sigma = covariance matrix

            Scipy logpdf function will calculate the P(X) if we give the input vector, mean and covariance

            In Naive Bayes we assume all the input variable is  independence so we take the covariance along diagonal only and this == the variance
            we use smoothing in the fit to avoid  singular covariance problem

            Methods:

            NaiveBayes.fit(X,Y)
            This method will store each class in a dict as key and the value is mean and variance for all input that in that class
            will calculate the prior of each class and store it in a dict

            NaiveBayes.predict(X)
            will calculate the logpdf for each class and store P, then return argmax at axis=1 np.argmax(axis=1)

            NaiveBayes.score(X,Y)
            # pass the X you want to predict or test and the actual value Y  ,after training the model using .fit
            # it will predict the value of X and then test if the predicted value == the acutal one and sum all the True and divid on the total number
            # at the end it will return the classification rate

    """

    def fit(self, X, Y, somoothing=10e-3):
        self.gaussians = dict() 	# creating empty dict for the gaussian parameters
        self.priors = dict()		# creating empty dict for the priors
        labels = set(Y)				# we use (set) to get all the unique value of y
        for c in labels:			# we loop through all the unique values
            # we set current_x to the  y == to the current label
            current_x = X[Y == c]
            self.gaussians[c] = {  # we store the mean and the variance for this gaussian
                'mean': current_x.mean(axis=0),
                'var': current_x.var(axis=0) + somoothing
            }
            # calculate the priors
            self.priors[c] = float(len(Y[Y == c])) / len(Y)
            # to be more efficient you can calculate the log prior

    def score(self, X, Y):
        P = self.predict(X)
        return np.mean(P == Y)

    def predict(self, X):
        N, D = X.shape
        k = len(self.gaussians)  # k == the number of classes
        # P is the predict we set it to the size of N,k (number of sample and
        # number of classes)
        P = np.zeros((N, k))
        # for each of the N sample we will calculate a k different probability
        for c, g in self.gaussians.items(): 	# we loop throw all the gaussian
            # get the mean and the varianece for this gaussian
            mean, var = g['mean'], g['var']
            P[:, c] = mvn.logpdf(X, mean=mean, cov=var) + \
                np.log(self.priors[c])
            # mvn.logpdf() can calculate the logpdf for multiple data point at
            # the same time
        # we got the argmax in axis =1 this will give us an N size array
        return np.argmax(P, axis=1)


if __name__ == '__main__':
    X, y = get_data(12000)
    #X,y = get_xor()
    #X,y = get_donut()
    Ntrain = int(len(y) / 2)
    X_train, y_train = X[:Ntrain], y[:Ntrain]
    X_test, y_test = X[Ntrain:], y[Ntrain:]

    classifier = NaiveBayes()
    t0 = datetime.now()
    classifier.fit(X_train, y_train)
    print(f"Training time is : {datetime.now()-t0}")

    t0 = datetime.now()
    print(f"Train accuracy is : {classifier.score(X_train,y_train)}")
    print(f"Time to compute train accuracy is {datetime.now()-t0}")

    t0 = datetime.now()
    print(f"Test accuracy is : {classifier.score(X_test,y_test)}")
    print(f"Time to compute Test accuracy is {datetime.now()-t0}")
