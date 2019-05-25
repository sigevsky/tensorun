import numpy as np


class MSE:
    def __init__(self):
        self._y_hat = None
        self._y = None

    def __call__(self, y, y_hat):
        assert y_hat.shape == y.shape
        self._y_hat = y_hat
        self._y = y
        return 1 / (2 * y.shape[1]) * np.sum(np.power((y - y_hat), 2))

    def backwards(self):
        return 1 / self._y_hat.shape[1] * (self._y_hat - self._y)


class CrossEntropyLoss:
    def __init__(self):
        self._y_hat = None
        self._y = None

    def __call__(self, y, y_hat):
        assert y_hat.shape == y.shape
        self._y_hat = y_hat
        self._y = y
        return -1 / y.size * np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

    def backwards(self):
        return -1 / self._y_hat.size * (self._y / self._y_hat - (1 - self._y) / (1 - self._y_hat))
