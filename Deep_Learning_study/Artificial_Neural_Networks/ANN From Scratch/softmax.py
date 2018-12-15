import numpy as np 
import maplotlib.pyplot as plt 
import pandas as pd 


def softmax(w,X):
	return np.exp(X.dot(w))/(np.exp(X.dot(w)))
