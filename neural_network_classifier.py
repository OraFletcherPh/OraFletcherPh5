import numpy as np

class NeuralNetworkClassifier:
    def __init__(self, input_size, hidden_layers, output_size):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.weights = self.initialize_weights()

    def initialize_weights(self):
        weights = []
        layer_sizes = [self.input_size] + self.hidden_layers + [self.output_size]

        for i in range(len(layer_sizes) - 1):
            weight_matrix = np.random.randn(layer_sizes[i], layer_sizes[i + 1])
            weights.append(weight_matrix)

        return weights

    def forward_propagation(self, X):
        activations = [X]
        for i in range(len(self.hidden_layers) + 1):
            activation = self.sigmoid(np.dot(activations[i], self.weights[i]))
            activations.append(activation)

        return activations

    def backward_propagation(self, X, y, activations, learning_rate):
        num_examples = X.shape[0]
        deltas = [None] * (len(self.hidden_layers) + 1)
        deltas[-1] = activations[-1] - y

        for i in range(len(self.hidden_layers), 0, -1):
            delta = np.dot(deltas[i], self.weights[i].T) * self.sigmoid_derivative(activations[i])
            deltas[i - 1] = delta

        for i in range(len(self.hidden_layers) + 1):
            self.weights[i] -= learning_rate * np.dot(activations[i].T, deltas[i]) / num_examples

    def train(self, X, y, num_epochs, learning_rate):
        for epoch in range(num_epochs):
            activations = self.forward_propagation(X)
            self.backward_propagation(X, y, activations, learning_rate)

    def predict(self, X):
        activations = self.forward_propagation(X)
        predictions = activations[-1]
        return np.argmax(predictions, axis=1)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))


# Example usage:
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Create a neural network classifier with 2 input nodes, 2 hidden layers with 4 nodes each,
# and 1 output node
classifier = NeuralNetworkClassifier(2, [4, 4], 1)

# Train the classifier
classifier.train(X, y, num_epochs=10000, learning_rate=0.1)

# Predict the output for the input
predictions = classifier.predict(X)
print("Predictions:", predictions)
