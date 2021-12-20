#! / usr / bin / env python3
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

sign = lambda x: (2 * (x >= 0) - 1) * (x != 0)


# Осуществляет чтение данных
def read_data(path="boston.csv"):
    dataframe = np.genfromtxt(path, delimiter=",", skip_header=15)
    np.random.seed(42)
    np.random.shuffle(dataframe)
    X = dataframe[:, :-1]
    y = dataframe[:, -1]
    return X, y


# Mean squared error
def mse(y_true, y_predicted):
    return sum((y_true - y_predicted) ** 2) / len(y_true)


# Коэффициент детерминации
def r2(y_true, y_predicted):
    u = sum((y_true - y_predicted) ** 2)
    v = sum((y_true - np.mean(y_true)) ** 2)
    return 1 - u / v


# Стандартная линейная регрессия
class NormalLR:
    def fit(self, X:np.ndarray, y:np.ndarray):
        n, k = X.shape
        X = np.insert(X, [k], [[1] for i in range(n)], 1)
        self.w = np.linalg.inv(np.dot(X.T, X)).dot(X.T).dot(y)
    
    def predict(self, X:np.ndarray) -> np.ndarray:
        return np.dot(X, self.w[:-1]) + self.w[-1]


# L1-регрессия, обучается с помощью градиентного спуска
class GradientLR:
    def __init__(self, alpha:float, iterations=10000, l=0.):
        self.alpha = alpha
        self.iterations = iterations
        self.l = l
        self.w = None
    
    def gradient(self, X:np.ndarray, y:np.ndarray):
        n, k = X.shape
        grad = np.zeros(k)
        for ind in range(k):
            for i in range(n):
                grad[ind] += X[i][ind] * (np.dot(self.w.T, X[i]) - y[i])
            grad[ind] *= 2
            grad[ind] += self.l * sign(self.w[ind])
        return grad
    
    def fit(self, X:np.ndarray, y:np.ndarray):
        n, k = X.shape
        X = np.insert(X, [k], [[1] for i in range(n)], 1)
        self.w = np.zeros(k + 1)
        for _ in range(self.iterations):
            self.w -= self.alpha * self.gradient(X, y)
        
    def predict(self, X:np.ndarray):
        return np.dot(X, self.w[:-1]) + self.w[-1]


# Строит график зависимости MSE на тестовой выборки от коэффициента L1-регрессии
def build_plot(X_train, y_train, X_test, y_test, left=0.0, right=0.5, step=0.001, l_rate=0.001, iterat=100):
    xs = np.arange(left, right + step, step)
    errors = []
    for x in xs:
        regr = GradientLR(l_rate, iterations=iterat, l=x)
        regr.fit(X_train, y_train)
        errors.append(mse(y_test, regr.predict(X_test)))
    plt.plot(xs, errors)
    plt.show()


# Чтение данных из датасета Бостон, разбиение на тестовую и тренировочную выборки
X, y = read_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=False)

# Стандартизация тренировочных данных
n, k = X.shape
m = np.zeros(k + 1); st = np.zeros(k + 1)
for i in range(k):
    m[i] = X_train[:, i].mean()
    X_train[:, i] -= m[i]
    st[i] = X_train[:, i].std()
    X_train[:, i] /= st[i]
m[k] = y_train.mean()
y_train -= m[k]
st[k] = y_train.std()
y_train /= st[k]

# Стандартизация тестовых данных
for i in range(k):
    X_test[:, i] -= m[i]
    X_test[:, i] /= st[i]
y_test -= m[k]
y_test /= st[k]

# Обучение обычной линейной регрессии и оценка её работы на тестовых данных
regr1 = NormalLR()
regr1.fit(X_train, y_train)
y_pred = regr1.predict(X_test)
print(f"NormalLR: \nMSE: {mse(y_test, y_pred)}, R2: {r2(y_test, y_pred)}")

# Обучение L1-регрессии и оценка её работы на тестовых данных
regr2 = GradientLR(0.0001, iterations=1000, l=0.2)
regr2.fit(X_train, y_train)
y_pred = regr2.predict(X_test)
print(f"\nGradientLR: \nMSE: {mse(y_test, y_pred)}, R2: {r2(y_test, y_pred)}")

# График зависимости MSE на тестовых данных от коэффициента L1-регрессии
build_plot(X_train, y_train, X_test, y_test, right=2, step=0.025, l_rate=0.0001)

# Результаты применения обычной линейной регрессии
print(f"\nNormalLR:")
names = ['Уровень преступности на душу населения',
         'Доля жилой земли, выделенной для участков площадью более 25000 кв. футов',
         'Доля акров на район, не связанных с розничной торговлей',
         'Наличие границы с рекой',
         'Концентрация оксидов азота',
         'Среднее количество комнат на жильё',
         'Доля жилых домов, построенных до 1940 года',
         'Взвешенное расстояние до пяти главных бостонских центров трудоустройства',
         'Индекс доступности радиальных магистралей',
         'Полная ставка налога на имущество (на 10 долларов США)',
         'Соотношение ученик/учитель на район',
         'Величина 1000*(Bk - 0.63)^2, где Bk- доля чернокожего населения на район',
         'Процент низкого статуса населения']
for i in range(len(names)):
    names[i] = (names[i], regr1.w[i])
sorted_names = sorted(names, key = lambda a: a[1])
for name, val in sorted_names:
    print(name, ':', val)
print(f'Ожидаемая стоимость жилья: {regr1.w[-1] * st[-1] + m[-1]}')

# Результаты применения L1-регрессии
print(f"\nGradientLR:")
names = ['Уровень преступности на душу населения',
         'Доля жилой земли, выделенной для участков площадью более 25000 кв. футов',
         'Доля акров на район, не связанных с розничной торговлей',
         'Наличие границы с рекой',
         'Концентрация оксидов азота',
         'Среднее количество комнат на жильё',
         'Доля жилых домов, построенных до 1940 года',
         'Взвешенное расстояние до пяти главных бостонских центров трудоустройства',
         'Индекс доступности радиальных магистралей',
         'Полная ставка налога на имущество (на 10 долларов США)',
         'Соотношение ученик/учитель на район',
         'Величина 1000*(Bk - 0.63)^2, где Bk- доля чернокожего населения на район',
         'Процент низкого статуса населения']
for i in range(len(names)):
    names[i] = (names[i], regr2.w[i])
sorted_names = sorted(names, key = lambda a: a[1])
for name, val in sorted_names:
    print(name, ':', val)
print(f'Ожидаемая стоимость жилья: {regr2.w[-1] * st[-1] + m[-1]}')


