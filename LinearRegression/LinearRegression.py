import numpy as np


class LinearRegression:
    def __init__(self, learning_rate=0.002, number_of_iterations=500):
        self.learning_rate = learning_rate
        self.number_of_iterations = number_of_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        number_of_samples, number_of_features = X.shape
        self.weights = np.zeros(number_of_features)
        self.bias = 0

        for _ in range(self.number_of_iterations):
            y_predicted = np.dot(X, self.weights) + self.bias

            # we are not doing the summation here because numpy.dot does it automatically
            dw = (1 / number_of_samples) * np.dot(X.T, (y_predicted - y))

            db = (1 / number_of_samples) * np.sum(y_predicted - y)

            self.weights = self.weights - self.learning_rate * dw
            self.bias = self.bias - self.learning_rate * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
