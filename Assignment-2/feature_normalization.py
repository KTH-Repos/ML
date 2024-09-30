import numpy as np

# Dataset
age = np.array([28, 35, 42, 25, 30])
income = np.array([60000, 70000, 80000, 55000, 65000])

# Calculate mean and standard deviation for each feature
age_mean = np.mean(age)
age_std = np.std(age)
income_mean = np.mean(income)
income_std = np.std(income)

# Perform Z-Score normalization
age_normalized = (age - age_mean) / age_std
income_normalized = (income - income_mean) / income_std

# Print the normalized data
print("Normalized Age:", age_normalized)
print("Normalized Income:", income_normalized)
