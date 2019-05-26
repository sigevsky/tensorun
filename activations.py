import numpy as np


class SigmaActivation:

    def __init__(self):
        self._cache = None

    @staticmethod
    def _sig(x):
        return 1. / (1. + np.power(np.e, -x))

    def forward(self, x):
        self._cache = self._sig(x)
        return self._cache

    def backwards(self, e):
        dsig = self._cache * (1 - self._cache)
        return e * dsig
