import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss

# Загрузка данных и разделение на обучающую и тестовую выборки
data = pd.read_csv('gbm-data.csv')
X = data.iloc[:, 1:].values  # Первая колонка - целевая переменная, остальные - признаки
y = data.iloc[:, 0].values  # Целевая переменная
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=241)

# Список для хранения значений log-loss на тестовой выборке
test_loss = []

# Обучение GradientBoostingClassifier с разными learning_rate
learning_rates = [1, 0.5, 0.3, 0.2, 0.1]
n_estimators = 250

for lr in learning_rates:
    clf = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=lr, verbose=True, random_state=241)
    clf.fit(X_train, y_train)

    # Предсказание вероятностей классов
    y_pred = clf.staged_decision_function(X_test)

    # Вычисление log-loss на тестовой выборке
    test_loss_lr = [log_loss(y_test, 1 / (1 + np.exp(-y_pred_i))) for y_pred_i in y_pred]
    test_loss.append(test_loss_lr)


min_loss_value = float('inf')
min_loss_iter = 0

for lr, loss in zip(learning_rates, test_loss):
    plt.plot(range(1, n_estimators + 1), loss, label=f'learning_rate={lr}')
    min_lr_loss = min(loss)
    if min_lr_loss < min_loss_value:
        min_loss_value = min_lr_loss
        min_loss_iter = np.argmin(loss) + 1

plt.legend(loc='upper left')
plt.xlabel('Iteration')
plt.ylabel('Log-Loss')
plt.title('Gradient Boosting Log-Loss')
plt.show()

# Ответ на вопрос 3
answer3 = "overfitting"  # График качества на тестовой выборке показывает признаки переобучения

# Ответ на вопрос 4
answer4 = f"{min_loss_value:.2f} {min_loss_iter}"

# Обучение RandomForestClassifier
rf_clf = RandomForestClassifier(n_estimators=min_loss_iter, random_state=241)
rf_clf.fit(X_train, y_train)

# Предсказание вероятностей классов
rf_y_pred = rf_clf.predict_proba(X_test)

# Вычисление log-loss на тестовой выборке для RandomForestClassifier
rf_loss = log_loss(y_test, rf_y_pred)

# Ответ на вопрос 5
answer5 = f"{rf_loss:.2f}"

print(f"Ответ на вопрос 3: {answer3}")
print(f"Ответ на вопрос 4: {answer4}")
print(f"Ответ на вопрос 5: {answer5}")