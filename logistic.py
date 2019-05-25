import numpy as np


class LogisticLayer:
    def __init__(self, dim):
        self.W = np.random.rand(1, dim)
        self.b = np.random.random()
        self._sig_f = None
        self._x_f = None

    @staticmethod
    def sig(x):
        return 1. / (1. + np.power(np.e, -x))

    def __call__(self, x):
        self._sig_f = self.sig(np.dot(self.W, x) + self.b)
        self._x_f = x
        return self._sig_f

    def backward(self, er):
        dsig = self._sig_f * (1 - self._sig_f)
        e = er * dsig  # shape = (1, m)
        dw = 1 / e.size * np.sum(np.dot(e, self._x_f.T), axis=0)
        db = 1 / e.size * np.sum(e)
        return dw, db
