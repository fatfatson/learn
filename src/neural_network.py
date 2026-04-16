"""
Neural Network from scratch using only NumPy.

Implements a Multi-Layer Perceptron (MLP) for MNIST digit classification.
Supports configurable architecture, ReLU activation, softmax output,
cross-entropy loss, and mini-batch SGD with momentum.
"""

import numpy as np


class NeuralNetwork:
    """A multi-layer perceptron implemented from scratch with NumPy."""

    def __init__(self, layer_sizes, learning_rate=0.1, momentum=0.9):
        """
        Initialize the neural network.

        Args:
            layer_sizes: List of integers, e.g. [784, 128, 64, 10]
            learning_rate: Learning rate for SGD
            momentum: Momentum coefficient for SGD
        """
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.num_layers = len(layer_sizes) - 1

        # Initialize weights with He initialization, biases with zeros
        self.weights = []
        self.biases = []
        self.vel_w = []  # velocity for momentum
        self.vel_b = []

        for i in range(self.num_layers):
            w = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(
                2.0 / layer_sizes[i]
            )
            b = np.zeros((1, layer_sizes[i + 1]))
            self.weights.append(w)
            self.biases.append(b)
            self.vel_w.append(np.zeros_like(w))
            self.vel_b.append(np.zeros_like(b))

    def relu(self, z):
        """ReLU activation function."""
        return np.maximum(0, z)

    def relu_derivative(self, z):
        """Derivative of ReLU."""
        return (z > 0).astype(float)

    def softmax(self, z):
        """Softmax activation with numerical stability."""
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def forward(self, X):
        """
        Forward pass through the network.

        Args:
            X: Input data, shape (batch_size, input_size)

        Returns:
            Output probabilities, shape (batch_size, output_size)
        """
        self.activations = [X]
        self.z_values = []

        a = X
        for i in range(self.num_layers):
            z = a @ self.weights[i] + self.biases[i]
            self.z_values.append(z)

            if i < self.num_layers - 1:
                # Hidden layers: ReLU
                a = self.relu(z)
            else:
                # Output layer: Softmax
                a = self.softmax(z)

            self.activations.append(a)

        return a

    def compute_loss(self, y_pred, y_true):
        """
        Compute cross-entropy loss.

        Args:
            y_pred: Predicted probabilities, shape (batch_size, num_classes)
            y_true: One-hot encoded true labels, shape (batch_size, num_classes)

        Returns:
            Scalar loss value
        """
        m = y_true.shape[0]
        # Clip to avoid log(0)
        y_pred_clipped = np.clip(y_pred, 1e-12, 1 - 1e-12)
        loss = -np.sum(y_true * np.log(y_pred_clipped)) / m
        return loss

    def backward(self, y_true):
        """
        Backward pass (backpropagation).

        Args:
            y_true: One-hot encoded true labels, shape (batch_size, num_classes)

        Returns:
            List of (weight_gradient, bias_gradient) for each layer
        """
        m = y_true.shape[0]
        grads_w = []
        grads_b = []

        # Output layer gradient (softmax + cross-entropy combined)
        delta = self.activations[-1] - y_true  # shape: (m, output_size)

        for i in range(self.num_layers - 1, -1, -1):
            dw = (self.activations[i].T @ delta) / m
            db = np.sum(delta, axis=0, keepdims=True) / m
            grads_w.insert(0, dw)
            grads_b.insert(0, db)

            if i > 0:
                delta = (delta @ self.weights[i].T) * self.relu_derivative(
                    self.z_values[i - 1]
                )

        return grads_w, grads_b

    def update_params(self, grads_w, grads_b):
        """Update weights and biases using SGD with momentum."""
        for i in range(self.num_layers):
            self.vel_w[i] = self.momentum * self.vel_w[i] - self.learning_rate * grads_w[i]
            self.vel_b[i] = self.momentum * self.vel_b[i] - self.learning_rate * grads_b[i]
            self.weights[i] += self.vel_w[i]
            self.biases[i] += self.vel_b[i]

    def train_step(self, X, y_true):
        """
        Perform one training step.

        Args:
            X: Input batch, shape (batch_size, input_size)
            y_true: One-hot encoded labels, shape (batch_size, num_classes)

        Returns:
            Loss value for this step
        """
        y_pred = self.forward(X)
        loss = self.compute_loss(y_pred, y_true)
        grads_w, grads_b = self.backward(y_true)
        self.update_params(grads_w, grads_b)
        return loss

    def predict(self, X):
        """
        Predict class labels.

        Args:
            X: Input data, shape (n_samples, input_size)

        Returns:
            Predicted class indices, shape (n_samples,)
        """
        probs = self.forward(X)
        return np.argmax(probs, axis=1)

    def predict_proba(self, X):
        """
        Predict class probabilities.

        Args:
            X: Input data, shape (n_samples, input_size)

        Returns:
            Class probabilities, shape (n_samples, num_classes)
        """
        return self.forward(X)

    def accuracy(self, X, y):
        """
        Compute accuracy on given data.

        Args:
            X: Input data, shape (n_samples, input_size)
            y: True labels (integer), shape (n_samples,)

        Returns:
            Accuracy as a float between 0 and 1
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)

    def save(self, filepath):
        """Save model parameters to a .npz file."""
        save_dict = {
            "layer_sizes": np.array(self.layer_sizes),
            "learning_rate": np.array(self.learning_rate),
            "momentum": np.array(self.momentum),
        }
        for i in range(self.num_layers):
            save_dict[f"w_{i}"] = self.weights[i]
            save_dict[f"b_{i}"] = self.biases[i]
        np.savez(filepath, **save_dict)
        print(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath):
        """Load model parameters from a .npz file."""
        data = np.load(filepath)
        layer_sizes = data["layer_sizes"].tolist()
        lr = float(data["learning_rate"])
        momentum = float(data["momentum"])

        model = cls(layer_sizes, learning_rate=lr, momentum=momentum)
        for i in range(model.num_layers):
            model.weights[i] = data[f"w_{i}"]
            model.biases[i] = data[f"b_{i}"]

        print(f"Model loaded from {filepath}")
        return model
