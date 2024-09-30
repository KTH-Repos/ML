import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Dataset
temperature = np.array([25, 30, 35, 20, 28])
humidity = np.array([50, 60, 70, 40, 55])
energy_consumption = np.array([200, 250, 300, 180, 220])

# Reshape data
temperature = temperature.reshape(-1, 1)
humidity = humidity.reshape(-1, 1)
energy_consumption = energy_consumption.reshape(-1, 1)

# Original feature set
X_original = np.hstack((temperature, humidity))

# New feature set with synthetic feature
temperature_humidity = (temperature * humidity).reshape(-1, 1)
X_new = np.hstack((X_original, temperature_humidity))

# Split the data into training and testing sets
X_train_original, X_test_original, y_train, y_test = train_test_split(X_original, energy_consumption, test_size=0.2, random_state=42)
X_train_new, X_test_new, y_train, y_test = train_test_split(X_new, energy_consumption, test_size=0.2, random_state=42)

# Create and train the model with original features
model_original = LinearRegression()
model_original.fit(X_train_original, y_train)

# Create and train the model with the new feature
model_new = LinearRegression()
model_new.fit(X_train_new, y_train)

# Make predictions with both models
y_pred_original = model_original.predict(X_test_original)
y_pred_new = model_new.predict(X_test_new)

# Evaluate the models using MSE
mse_original = mean_squared_error(y_test, y_pred_original)
mse_new = mean_squared_error(y_test, y_pred_new)

print(f"Mean Squared Error with original features: {mse_original}")
print(f"Mean Squared Error with new feature: {mse_new}")
