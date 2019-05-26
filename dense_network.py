import numpy as np
from dense import Dense
from activations import SigmaActivation
from losses import MSE


class DenseNode:
    def __init__(self, layer: Dense, activation: SigmaActivation):
        self.layer = layer
        self.activation = activation


class TrivialGradientOptimizer:
    def __init__(self, alpha: np.float):
        self.alpha = alpha


class SimpleDenseNetwork:

    def __init__(self, *nodes: DenseNode, optimizer: TrivialGradientOptimizer, loss: MSE):
        self.nodes = nodes
        self.optimizer = optimizer
        self.loss = loss

    def forward(self, x):
        for node in self.nodes:
            x = node.activation.forward(node.layer.forward(x))

        return x

    def backwards(self, y_hat, y):
        ls = self.loss(y_hat, y)
        e = self.loss.backwards()
        for node in reversed(self.nodes):
            act, layer = node.activation, node.layer
            e, dW, db = layer.backwards(act.backwards(e))
            layer.W = layer.W - self.optimizer.alpha * dW
            layer.b = layer.b - self.optimizer.alpha * db

        return ls

