import numpy as np

def last_col(data):
    return data[:, len(data) - 1]


def init_col(data):
    return data[:, 0:len(data) - 1]


def last_as_label(data):
    return [(i[0:-1], i[-1]) for i in data]


def join_data(x, y):
    return [(x[i], y[i]) for i in range(0, len(x))]


def join_np_data(X, Y):
    return np.vstack((X, Y)).T
