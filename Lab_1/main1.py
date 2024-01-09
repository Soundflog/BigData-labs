import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge
from scipy.sparse import hstack


def Refactoring_csv_file(fileName, dtTest=False):
    dt = pd.read_csv(fileName)
    # Привести тексты к нижнему регистру и заменить всё, кроме букв и цифр, на пробелы
    dt['FullDescription'] = (dt['FullDescription']
                             .apply(lambda x: re.sub('[^a-zA-Z0-9]', ' ', x.lower())))

    if not dtTest:
        # Замена пропусков на 'nan' в столбце 'LocationNormalized'
        dt['LocationNormalized'].fillna('nan', inplace=True)

        # Замена пропусков на 'nan' в столбце 'ContractTime'
        dt['ContractTime'].fillna('nan', inplace=True)
    return dt


def Regression(target, features):
    # Обучить гребневую регрессию
    clf = Ridge(alpha=1.0)
    clf.fit(features, target)
    return clf


if __name__ == '__main__':
    # Загрузить данные из файла salary-train.csv
    data_train = Refactoring_csv_file('salary-train.csv')
    # Инициализация векторов
    tfidf_vectorizer = TfidfVectorizer(min_df=5)
    enc = DictVectorizer()

    # Преобразование текстовых данных в векторы признаков на обучающей выборке
    tfidf_features = tfidf_vectorizer.fit_transform(data_train['FullDescription'])
    cat_features = enc.fit_transform(data_train[['LocationNormalized', 'ContractTime']].to_dict('records'))
    # Объединить векторы признаков
    df_train_features = hstack([tfidf_features, cat_features])
    # Обучение на гребневой регрессии обучающей выборкой
    ridge_regression = Regression(data_train['SalaryNormalized'], df_train_features)

    # Загрузить данные из файла salary-test-mini.csv
    data_test = Refactoring_csv_file('salary-test-mini.csv', True)

    # Преобразование текстовых данных в векторы признаков на тестовых данных
    tfidf_test_features = tfidf_vectorizer.transform(data_test['FullDescription'])
    cat_test_features = enc.transform(data_test[['LocationNormalized', 'ContractTime']].to_dict('records'))
    # Объединить векторы признаков
    df_test_features = hstack([tfidf_test_features, cat_test_features])

    # Построить прогнозы
    predictions = ridge_regression.predict(df_test_features)
    print(predictions)
