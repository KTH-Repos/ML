import numpy as np
from sklearn.cluster import KMeans

# Dataset
data = np.array([
    [2.0, 1.0],
    [2.5, 2.2],
    [1.8, 1.8],
    [5.0, 6.0],
    [5.5, 7.0],
    [4.5, 5.5]
])

# Initial cluster centers
initial_centers = np.array([
    [4.75, 5.75],
    [2.1, 1.67],
    [5.5, 7]
])

# Perform k-means clustering with specified initial centers
kmeans = KMeans(n_clusters=3, init=initial_centers)
kmeans.fit(data)

cost = kmeans.inertia_

# Print the final cluster centers
print("Final cluster centers:")
print(kmeans.cluster_centers_)

# Print the labels assigned to each data point
print("\nLabels assigned to each data point:")
print(kmeans.labels_)

print("\nCost function (inertia):")
print(cost)