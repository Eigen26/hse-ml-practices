import numpy as np


def mse(y_true, y_predicted):
    return sum((y_true - y_predicted) ** 2) / len(y_true)


def r2(y_true, y_predicted):
    u = sum((y_true - y_predicted) ** 2)
    v = sum((y_true - np.mean(y_true)) ** 2)
    return 1 - u / v
