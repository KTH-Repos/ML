import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Define the dataset
data = np.array([
    [25, 120, 0],
    [30, 110, 1],
    [35, 130, 0],
    [40, 140, 1],
    [45, 115, 0]
])

# Separate features and labels
X = data[:, :-1]  # Features: Age, Blood Pressure
y = data[:, -1]   # Target: Condition

# Define the new patient
new_patient = np.array([[32, 125]])

# Initialize the KNN classifier with K=3
knn = KNeighborsClassifier(n_neighbors=3)

# Train the classifier
knn.fit(X, y)

# Predict the class for the new patient
predicted_class = knn.predict(new_patient)

# Print the result
print(f"The new patient with Age=32 and Blood Pressure=125 is classified as: {'Condition 1' if predicted_class[0] == 1 else 'Condition 0'}")
