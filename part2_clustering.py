import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.datasets import load_wine
wine = load_wine()
# wine object is a sklearn.utils.bunch object, need to convert in pandas dataframe.

# This next 2 lines of code is taken from
# "https://stackoverflow.com/questions/38105539/how-to-convert-a-scikit-learn-dataset-to-a-pandas-dataset"
data = pd.DataFrame(wine.data,columns=wine.feature_names)
data['class'] = pd.Series(wine.target)

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Splitting data into train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# Using StandardScaler to preprocess data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# REFERENCES FOR CLUSTERING
# https://www.superdatascience.com/pages/machine-learning
# https://www.linkedin.com/pulse/finding-optimal-number-clusters-k-means-through-elbow-asanka-perera
# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html
# https://medium.com/@sametgirgin/hierarchical-clustering-model-in-5-steps-with-python-6c45087d4318

# K-means clustering
from sklearn.cluster import KMeans

# Within Cluster Sum of Squares (WCSS)
wcss = []
for i in range(1, 10):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_train)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 10), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.ion()
plt.show()
plt.pause(2)
plt.close()

# Training the K-Means model on the dataset
print("\n Optimal number of clusters as seen by elbow method are 3 clusters.\n")
kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X_train)

plt.scatter(X_train[:, 0], X_train[:, 1], c=y_kmeans, s=50, cmap='viridis')

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.ion()
plt.show()
plt.pause(2)
plt.close()

from sklearn.metrics.cluster import homogeneity_score,completeness_score, v_measure_score
print("------------------------------------------------------------------------------------")
print("-Printing Homogeneity_score, Completeness_score, v_measure_score for K-means method-")
print("------------------------------------------------------------------------------------")
print("Homogeneity_score :",homogeneity_score(y_train,y_kmeans))
print("Completeness_score :",completeness_score(y_train,y_kmeans))
print("v_measure_score :",v_measure_score(y_train,y_kmeans))
print("\n")

# Hierarchy clustering
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X_train, method  = "ward"))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.ion()
plt.show()
plt.pause(2)
plt.close()

print("\n Optimal number of clusters from hierarchical clustering graph can be seen as 3 clusters.\n")
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 3, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X_train)
print("-----------------------------------------------------------------------------------------")
print("-Printing Homogeneity_score, Completeness_score, v_measure_score for Hierarchical method-")
print("-----------------------------------------------------------------------------------------")
print("Homogeneity_score :",homogeneity_score(y_train,y_hc))
print("Completeness_score :",completeness_score(y_train,y_hc))
print("v_measure_score :",v_measure_score(y_train,y_hc))


