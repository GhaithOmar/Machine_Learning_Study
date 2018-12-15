import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

df = pd.read_csv('Mall_Customers.csv')
X = df.iloc[:,3:].values

# Using the dendrogram to find the optimal number of clusters

import scipy.cluster.hierarchy as  sch 

dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))

plt.title('Dendrogram')
plt.xlabel("Customers")
plt.ylabel('Euclidean distances')
plt.show()

# Fitting Hierarchical clustering to the mall dataset
from sklearn.cluster import AgglomerativeClustering

hc = AgglomerativeClustering(n_clusters= 5, affinity = 'euclidean',linkage= 'ward')
y_hc = hc.fit_predict(X)

# Visualising the clusters

plt.scatter(X[:,0],X[:,1], s=100, c=y_hc)
plt.show()