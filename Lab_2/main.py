import pandas as pd
from sklearn.decomposition import PCA
import numpy as np


def get_data(data):
    del data['date']
    return data


data_close_prices = pd.read_csv('close_prices.csv')
print(data_close_prices.head())
data_close_prices = get_data(data_close_prices)

model = PCA(n_components=10)
model.fit(data_close_prices)
print(f"Совокупная сумма всех процентов дисперсий --- {np.cumsum(model.explained_variance_ratio_)}")
print(f"90% дисперсии --- {np.argmax(np.cumsum(model.explained_variance_ratio_) >= 0.9)+1}\n")

print(f"Процент дисперсии --- {model.explained_variance_ratio_} \n")
print(f"компоненты --- {model.components_}\n")
transformed = model.transform(data_close_prices)
print(f" Первая компонента --- {transformed[:, 0]}\n")


data_djia_index = pd.read_csv('djia_index.csv')
djia_index = data_djia_index['^DJI']
print(f"DJIA index:\n {djia_index.head()}\n")


corrcoef = np.corrcoef(transformed[:, 0], djia_index)[0,1]
print(f"Корреляция ---- {corrcoef:.2f}")

weight = model.components_[0]
max_weight = np.argmax(weight)

print(f"Index компании с максимальным весос в 1 компоненте --- {max_weight}\n")
print(f"Компания с наибольшим весом в первой компоненте --- {data_close_prices.columns[max_weight]}\n")
print(f"Вес компании --- {weight[max_weight]}")
