import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Sample data: hours studied and corresponding grades
hours_studied = np.array([2, 3, 4, 5, 6, 7, 8])
grades_obtained = np.array([68, 75, 83, 89, 92, 95, 98])

# Reshape data
hours_studied = hours_studied.reshape(-1, 1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(hours_studied, grades_obtained, test_size=0.2, random_state=42)

# Create the model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Find the intercept and slope
intercept = model.intercept_
slope = model.coef_[0]

print(f"Intercept: {intercept}")
print(f"Slope: {slope}")
print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# Plot the results
plt.scatter(hours_studied, grades_obtained, color='blue', label='Actual Data')
plt.plot(hours_studied, model.predict(hours_studied), color='red', label='Regression Line')
plt.xlabel('Hours Studied')
plt.ylabel('Grade Obtained')
plt.legend()
plt.title('Hours Studied vs Grade Obtained')
plt.show()