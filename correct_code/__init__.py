'''
Исследуем зависимость стоимости жилья от его характеристик с помощью 2 моделей:
обычной модели лин. регрессии и модели регрессии "лассо" (с L1-регуляризацией).
'''
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from linear_regression import NormalLR
from lasso_regression import GradientLR
from metrics import mse, r2


def read_data(path="boston.csv"):
    '''Осуществляет чтение данных'''
    dataframe = np.genfromtxt(path, delimiter=",", skip_header=15)
    np.random.seed(42)
    np.random.shuffle(dataframe)
    X = dataframe[:, :-1]
    y = dataframe[:, -1]
    return X, y


def build_plot(
    X_train, y_train,
    X_test, y_test,
    left=0.0, right=0.5,
    st=0.001, rate=0.001, it=100
):
    '''Строит график зав-ти MSE (на тест. выборке) от коэф. L1-регрессии'''
    xs = np.arange(left, right + st, st)
    errors = []
    for x in xs:
        regr = GradientLR(rate, iterations=it, coef=x)
        regr.fit(X_train, y_train)
        errors.append(mse(y_test, regr.predict(X_test)))
    plt.plot(xs, errors)
    plt.show()


if __name__ == '__main__':
    # Чтение данных из датасета Бостон, разбиение на тестовую и трен. выборки
    X, y = read_data()
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, train_size=0.8, shuffle=False)

    # Стандартизация тренировочных данных
    n, k = X.shape
    m = np.zeros(k + 1)
    st = np.zeros(k + 1)
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
    regr2 = GradientLR(0.0001, iterations=1000, coef=0.2)
    regr2.fit(X_train, y_train)
    y_pred = regr2.predict(X_test)
    print(
        f"\nGradientLR: \nMSE: {mse(y_test, y_pred)}, R2: {r2(y_test, y_pred)}"
        )

    # График зависимости MSE на тестовых данных от коэффициента L1-регрессии
    build_plot(X_train, y_train, X_test, y_test, right=2, st=0.025, rate=0.001)

    # Результаты применения обычной линейной регрессии
    print("\nNormalLR:")
    names = ['Уровень преступности на душу населения',
             'Доля жилой земли, выделенной для участков S > 25000 футов^2',
             'Доля акров на район, не связанных с розничной торговлей',
             'Наличие границы с рекой',
             'Концентрация оксидов азота',
             'Среднее количество комнат на жильё',
             'Доля жилых домов, построенных до 1940 года',
             'Взвеш. расст. до 5 глав. бостонских центров трудоустройства',
             'Индекс доступности радиальных магистралей',
             'Полная ставка налога на имущество (на 10 долларов США)',
             'Соотношение ученик/учитель на район',
             'Вел. 1000*(Bk - 0.63)^2, где Bk- доля чернокож. насел. на район',
             'Процент низкого статуса населения']
    for i in range(len(names)):
        names[i] = (names[i], regr1.w[i])
    sorted_names = sorted(names, key=lambda a: a[1])
    for name, val in sorted_names:
        print(name, ':', val)
    print(f'Ожидаемая стоимость жилья: {regr1.w[-1] * st[-1] + m[-1]}')

    # Результаты применения L1-регрессии
    print("\nGradientLR:")
    names = ['Уровень преступности на душу населения',
             'Доля жилой земли, выделенной для участков S > 25000 футов^2',
             'Доля акров на район, не связанных с розничной торговлей',
             'Наличие границы с рекой',
             'Концентрация оксидов азота',
             'Среднее количество комнат на жильё',
             'Доля жилых домов, построенных до 1940 года',
             'Взвеш. расст. до 5 глав. бостонских центров трудоустройства',
             'Индекс доступности радиальных магистралей',
             'Полная ставка налога на имущество (на 10 долларов США)',
             'Соотношение ученик/учитель на район',
             'Вел. 1000*(Bk - 0.63)^2, где Bk- доля чернокож. насел. на район',
             'Процент низкого статуса населения']
    for i in range(len(names)):
        names[i] = (names[i], regr2.w[i])
    sorted_names = sorted(names, key=lambda a: a[1])
    for name, val in sorted_names:
        print(name, ':', val)
    print(f'Ожидаемая стоимость жилья: {regr2.w[-1] * st[-1] + m[-1]}')
