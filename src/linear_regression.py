import numpy as np


class NormalLR:
    def fit(self, X:np.ndarray, y:np.ndarray):
        n, k = X.shape
        X = np.insert(X, [k], [[1] for i in range(n)], 1)
        self.w = np.linalg.inv(np.dot(X.T, X)).dot(X.T).dot(y)
    
    def predict(self, X:np.ndarray) -> np.ndarray:
        return np.dot(X, self.w[:-1]) + self.w[-1]
