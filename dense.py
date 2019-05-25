import numpy as np


class Dense:

    def __init__(self, n, m):
        self.W = np.random.rand(n, m)
        self.b = np.random.rand(n, 1)
        self._x_f = None  # (m x r)

    def forward(self, x):
        self._x_f = x
        return np.dot(self.W, x) + self.b

    def backward_update(self, e: np.ndarray):
        dW = np.dot(e, self._x_f.T)
        db = np.sum(e, axis=1, keepdims=True)

        # e: n * r  | W: n * m - transpose so as not to mess with (1 x n * r) dot (n * r x m * r) = (1 x m * r)
        # and propagate only rightes pair of the tuple
        return np.dot(self.W.T, e), dW, db
