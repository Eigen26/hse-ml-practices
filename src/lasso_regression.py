import numpy as np


def sign(x):
    return (2 * (x > 0) - 1) * (x != 0)


class GradientLR:
    def __init__(self, alpha: float, iterations=10000, coef=0.):
        self.alpha = alpha
        self.iterations = iterations
        self.coef = coef
        self.w = None

    def gradient(self, X: np.ndarray, y: np.ndarray):
        '''Считает текущее значение градиента'''
        n, k = X.shape
        grad = np.zeros(k)
        for ind in range(k):
            for i in range(n):
                grad[ind] += X[i][ind] * (np.dot(self.w.T, X[i]) - y[i])
            grad[ind] *= 2
            grad[ind] += self.coef * sign(self.w[ind])
        return grad

    def fit(self, X: np.ndarray, y: np.ndarray):
        '''Обучает модель, обновляя её веса с помощью градиентного спуска'''
        n, k = X.shape
        X = np.insert(X, [k], [[1] for i in range(n)], 1)
        self.w = np.zeros(k + 1)
        for _ in range(self.iterations):
            self.w -= self.alpha * self.gradient(X, y)

    def predict(self, X: np.ndarray):
        '''Предсказывает значение переменной y с помощью LASSO-регрессии'''
        return np.dot(X, self.w[:-1]) + self.w[-1]
