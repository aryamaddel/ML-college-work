# /// script
# dependencies = ["pandas", "matplotlib", "scikit-learn", "numpy"]
# ///

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Generate Synthetic Clustering Data (Blobs)
# n_samples=300, centers=4, cluster_std=0.60, random_state=42
X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=42)

df = pd.DataFrame(X, columns=["Feature_1", "Feature_2"])
df["True_Cluster"] = y

# Visualize Synthetic Data
plt.scatter(X[:, 0], X[:, 1], s=50, c="gray")
plt.title("Synthetic Clustering Data")
plt.show()

# Elbow Method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(
        n_clusters=i, init="k-means++", max_iter=300, n_init=10, random_state=42
    )
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title("Elbow Method")
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.show()

# K-Means (Visual performance)
kmeans = KMeans(
    n_clusters=4, init="k-means++", max_iter=300, n_init=10, random_state=42
)
y_kmeans = kmeans.fit_predict(X)

# Visualize Clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=50, c="red", label="Cluster 1")
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=50, c="blue", label="Cluster 2")
plt.scatter(
    X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s=50, c="green", label="Cluster 3"
)
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s=50, c="cyan", label="Cluster 4")
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    s=200,
    c="yellow",
    label="Centroids",
)
plt.title("K-Means Clustering on Synthetic Blobs")
plt.legend()
plt.show()
