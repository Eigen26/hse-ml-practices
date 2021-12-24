import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error as true_mse, r2_score as true_r2
from sklearn.model_selection import train_test_split
from linear_regression import NormalLR
from lasso_regression import GradientLR
from metrics import mse, r2


models = {'linear': (LinearRegression, NormalLR), 'lasso': (Lasso, GradientLR)}
metrics = {'mse': (true_mse, mse), 'r2': (true_r2, r2)}


def gen(n, xs=True):
    if xs:
        for i in range(n):
            yield np.random.uniform(low=-10, high=10, size=(100, 2)), np.random.uniform(low=0, high=1, size=100)
    else:
        for i in range(n):
            yield np.random.uniform(low=-100000, high=100000, size=100), np.random.uniform(low=-100000, high=100000, size=100)


@pytest.mark.parametrize('model', ['linear', 'lasso'])
@pytest.mark.parametrize('X, y', gen(300))
def test_regression(model, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
    LR1 = models[model][0](); LR1.fit(X_train, y_train)
    LR2 = models[model][0](); LR2.fit(X_train, y_train)
    pred1 = LR1.predict(X_test)
    pred2 = LR2.predict(X_test)
    diff = 1 - sum((pred1 - pred2) ** 2 < 0.001) / len(pred1)
    assert diff < 0.0001


@pytest.mark.parametrize('metric', ['mse', 'r2'])
@pytest.mark.parametrize('y1, y2', gen(200, xs=False))
def test_metrics(metric, y1, y2):
    res1 = metrics[metric][0](y1, y2)
    res2 = metrics[metric][1](y1, y2)
    assert round(res1, 3) == round(res2, 3)
