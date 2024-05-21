import numpy as np


class LinearRegression:
    def __init__(self, lr=0.001, n_iters=1000):
        """
        Initializes a new instance of the LinearRegression class.

        Parameters:
            lr (float, optional): The learning rate for gradient descent. Defaults to 0.001.
            n_iters (int, optional): The number of iterations for gradient descent. Defaults to 1000.

        Returns:
            None
        """
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            y_pred = np.dot(X, self.weights) + self.bias

            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db

    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred


# Define a function for reporting Mean Squared Error
def MSE(y_test, predictions):
    """
    Calculate the Mean Squared Error (MSE) between the actual values `y_test` and the predicted values `predictions`.

    Parameters:
        y_test (array-like): The actual values.
        predictions (array-like): The predicted values.

    Returns:
        float: The MSE value.
    """
    return np.mean((y_test - predictions) ** 2)
