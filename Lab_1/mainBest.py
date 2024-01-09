import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from scipy.sparse import hstack
from sklearn.linear_model import Ridge

# 1) Загрузить данные об описаниях вакансий и соответствующих годовых 
# зарплатах из файла salary-train.csv.
data_train = pd.read_csv('salary-train.csv')
data_test_mini = pd.read_csv('salary-test-mini.csv')

# 2) Провести предобработку:
# а) привести тексты к нижнему регистру;
# б) заменить всё, кроме букв и цифр, на пробелы – это облегчит дальнейшее
# разделение текста на слова. Для такой замены в строке text подходит
# следующий вызов:
# re.sub(’[^a–zA–Z0–9]’, ’␣’, text.lower())
# в) применить TfidfVectorizer для преобразования текстов в векторы
# признаков. Оставьте только те слова, которые встречаются хотя бы в 5
# объектах (параметр min_df у TfidfVectorizer);
# г) заменить пропуски в столбцах LocationNormalized и ContractTime на специальную строку
# ’nan’. Код для этого был приведен выше;
# д) применить DictVectorizer для получения one-hot-кодирования признаков LocationNormalized и
# ContractTime;
# е) объединить все полученные признаки в одну матрицу "объекты- признаки".
# Обратите внимание, что матрицы для текстов и категориальных признаков являются разреженными.
# Для объединения их столбцов нужно воспользоваться функцией scipy.sparse.hstack

# Приведение текстов к нижнему регистру
data_train['FullDescription'] = data_train['FullDescription'].str.lower()
data_test_mini['FullDescription'] = data_test_mini['FullDescription'].str.lower()

# Приведение текстов к нижнему регистру
data_train['FullDescription'] = data_train['FullDescription'].apply(lambda x: re.sub('[^a-zA-Z0-9]', ' ', x))
data_test_mini['FullDescription'] = data_test_mini['FullDescription'].apply(lambda x: re.sub('[^a-zA-Z0-9]', ' ', x))

# Применение TfidfVectorizer для текстов
tfidf_vectorizer = TfidfVectorizer(min_df=5)
X_train_tfidf = tfidf_vectorizer.fit_transform(data_train['FullDescription'])
X_test_mini_tfidf = tfidf_vectorizer.transform(data_test_mini['FullDescription'])

# Замена пропусков в категориальных признаках на 'nan'
data_train['LocationNormalized'].fillna('nan', inplace=True)
data_train['ContractTime'].fillna('nan', inplace=True)

# Применение DictVectorizer для one-hot-кодирования категориальных признаков
enc = DictVectorizer()
X_train_categ = enc.fit_transform(data_train[['LocationNormalized', 'ContractTime']].to_dict('records'))
X_test_mini_categ = enc.transform(data_test_mini[['LocationNormalized', 'ContractTime']].to_dict('records'))

# Применение DictVectorizer для one-hot-кодирования категориальных признаков
X_train = hstack([X_train_tfidf, X_train_categ])
X_test_mini = hstack([X_test_mini_tfidf, X_test_mini_categ])

# 3) Обучить гребневую регрессию с параметром alpha=1. Целевая 
# переменная записана в столбце SalaryNormalized
ridge = Ridge(alpha=1)
ridge.fit(X_train, data_train['SalaryNormalized'])

# 4) Построить прогнозы для двух примеров из файла salary-test-mini.csv.
# Значения полученных прогнозов являются ответом на задание. Укажите
# их через пробел
predictions = ridge.predict(X_test_mini)

# Вывод результатов
print(predictions)
