import numpy as np

# Given normalized data point
x = np.array([0.15, -0.66, 1.58, -1.17, 0.96, -0.86])

# Given eigenvectors corresponding to the first three principal components
v1 = np.array([0.35, 0.5, 0.45, 0.4, 0.2, 0.4])
v2 = np.array([-0.4, 0.2, -0.1, 0.6, -0.45, 0.45])
v3 = np.array([-0.1, -0.1, -0.3, -0.2, 0.8, -0.4])

# Create the eigenvector matrix (each column is an eigenvector)
V = np.column_stack((v1, v2, v3))

# Project the data point onto the new basis
x_pca = np.dot(x, V)

print("Projected data point (PCA):")
print(x_pca)


# Given eigenvalues
eigenvalues = np.array([0.3775, 0.0511, 0.0279, 0.0230, 0.0168, 0.0120, 0.0085, 0.0039, 0.0018])

# Calculate total variance
total_variance = np.sum(eigenvalues)

# Calculate cumulative variance
cumulative_variance = np.cumsum(eigenvalues)

# Calculate the variance required to keep at least 90% of the total variance
variance_threshold = 0.90 * total_variance

# Find the number of components needed to reach this variance threshold
num_components = np.argmax(cumulative_variance >= variance_threshold) + 1

print(f"Total variance: {total_variance:.4f}")
print(f"Cumulative variance: {cumulative_variance}")
print(f"Variance threshold for 90%: {variance_threshold:.4f}")
print(f"Minimum number of principal components to keep at least 90% variance: {num_components}")

