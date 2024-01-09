import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

# Загрузка данных
data = pd.read_csv('abalone.csv')

# Преобразование признака Sex в числовой
data['Sex'] = data['Sex'].map({'M': 1, 'F': -1, 'I': 0})

# Разделение на признаки и целевую переменную
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Список для хранения коэффициентов детерминации
r2_scores = []

# Перебор количества деревьев
for n_estimators in range(1, 51):
    # Создание модели случайного леса
    clf = RandomForestRegressor(n_estimators=n_estimators, random_state=1)

    # Создание генератора кросс-валидации
    cv = KFold(n_splits=5, shuffle=True, random_state=1)

    # Вычисление коэффициента детерминации на кросс-валидации
    r2 = cross_val_score(clf, X, y, cv=cv, scoring='r2')

    # Усреднение результатов кросс-валидации
    mean_r2 = np.mean(r2)
    r2_scores.append(mean_r2)

    # Печать результатов
    print(f"n_estimators = {n_estimators}, R2 = {mean_r2}")

# Поиск минимального количества деревьев с R2 > 0.52
min_estimators = next(i for i, r2 in enumerate(r2_scores) if r2 > 0.52)

print(f"Минимальное количество деревьев с R2 > 0.52: {min_estimators}")

plt.plot(range(1, 51), r2_scores)
plt.xlabel('Количество деревьев')
plt.ylabel('R2')
plt.title('Изменение качества случайного леса')
plt.show()
