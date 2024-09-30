import numpy as np

# Dataset
data = np.array([(1, 3), (2, 5), (3, 7), (4, 9), (5, 11)])
X = data[:, 0]
y = data[:, 1]

# Parameters
w0 = 0
w1 = 0
alpha = 0.01
iterations = 2
m = len(y)

# Gradient Descent
for _ in range(iterations):
    # Predictions
    y_pred = w0 + w1 * X

    # Calculate gradients
    dw0 = (1/m) * np.sum(y_pred - y)
    dw1 = (1/m) * np.sum((y_pred - y) * X)

    # Update weights
    w0 = w0 - alpha * dw0
    w1 = w1 - alpha * dw1

    print(f"Iteration {_+1}: w0 = {w0}, w1 = {w1}")

# Final weights after 2 iterations
print(f"Final weights: w0 = {w0}, w1 = {w1}")
