import numpy as np

# Define sigmoid activation function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Define the weights and biases
W1 = np.array([[0.2, 0.3], [0.4, -0.5], [-0.1, 0.2]])
b1 = np.array([[-0.1], [0.2], [0.3]])
W2 = np.array([[0.4, 0.1, 0.6]])
b2 = np.array([[0.2]])

# Define the input vector
x = np.array([[0.6], [0.8]])

# Forward propagation
# Hidden layer input
Z1 = np.dot(W1, x) + b1

# Hidden layer activation
A1 = sigmoid(Z1)

# Output layer input
Z2 = np.dot(W2, A1) + b2

# Output
output = Z2

print("Output of the network:", output)
