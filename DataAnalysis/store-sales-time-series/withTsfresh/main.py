import numpy as np
import pandas as pd
import datetime
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_log_error

from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from statsmodels.graphics.tsaplots import plot_pacf

train = pd.read_csv('/kaggle/input/store-sales-time-series-forecasting/train.csv', index_col='id', parse_dates=['date'])
test = pd.read_csv('/kaggle/input/store-sales-time-series-forecasting/test.csv', index_col='id', parse_dates=['date'])

# Объединить обучающий и тестовый наборы данных для разработки объектов, как для удобства, так и для учета
# переноса некоторых временных вложений объектов из более позднего периода обучающего набора данных в
# тестовый набор данных.
train_test = pd.concat([train, test], ignore_index=True)

# Чтобы изменить, преобразуйте столбец даты в тип данных datetime,
# затем выделите некоторые основные элементы календаря в виде отдельных функций.
train_test.date = pd.to_datetime(train_test.date)

train_test['year'] = train_test.date.dt.year
train_test['month'] = train_test.date.dt.month
train_test['dayofmonth'] = train_test.date.dt.day
train_test['dayofweek'] = train_test.date.dt.dayofweek
train_test['dayname'] = train_test.date.dt.strftime('%A')

oil = pd.read_csv('/kaggle/input/store-sales-time-series-forecasting/oil.csv', parse_dates=['date'], index_col='date').to_period('D')

# Чтобы создать скользящее среднее за 7 дней в качестве основы для поиска тенденций в нефти, создайте отдельный фрейм данных для изучения этих тенденций, а затем перенаправьте недостающие данные с начала значений скользящего среднего за 7 дней. Также попробовал скользящее среднее за 28 дней для нефти, но это увеличило баллы прогнозирования.
oil['avg_oil_7'] = oil['dcoilwtico'].rolling(7).mean()

trends = pd.DataFrame(index=pd.date_range('2013-01-01', '2017-08-31')).to_period('D')
trends = trends.join(oil, how='outer')
trends['avg_oil_7'].fillna(method='ffill', inplace=True)
trends.dropna(inplace=True)

# Автокорреляция рассматривает корреляцию значений временного ряда за последовательные периоды. Частичная автокорреляция устраняет любую косвенную корреляцию, которая может присутствовать. Если взять скользящее среднее за 7 дней для нефти и спроецировать его на 12 дней вперед, то первые три запаздывания оказывают наибольшее влияние. Таким образом, к общим характеристикам добавляются 3 задержки.
_ = plot_pacf(trends.avg_oil_7, lags=12)

n_lags = 3
for lag in range(1, n_lags + 1):
    trends[f'oil_lags7_{lag}'] = trends.avg_oil_7.shift(lag)
trends.dropna(inplace=True)

trends['date_str'] = trends.index.astype(str)
trends.drop('dcoilwtico', axis=1, inplace=True)

# В первый день отсутствуют данные о нефти, поэтому они заполняются тем же значением, что и на следующий день.
#
# Существует также ряд дней, в основном выходных, когда данных о нефти нет.
# Эти даты идентифицируются и затем заполняются линейным методом. Этот метод, по сути, рисует линию между двумя наблюдаемыми точками, а затем заполняет недостающие значения так, чтобы они лежали на этой линии.
oil = oil.interpolate(method='linear')
oil.iloc[0] = oil.iloc[1]

start_date = train_test.date.min()
number_of_days = 1704
date_list = [(start_date + datetime.timedelta(days=day)).isoformat() for day in range(number_of_days)]

date = (pd.Series(date_list)).to_frame()
date.columns = ['date']
date.date = pd.to_datetime(date.date)
date['date_str'] = date.date.astype(str)
oil['date_str'] = oil.index.astype(str)

oil = pd.merge(date, oil, how='left', on='date_str')
oil = oil.set_index('date').dcoilwtico.interpolate(method='linear').to_frame()
oil['date_str'] = oil.index.astype(str)

# Чтобы добавить данные о нефти и тенденциях в основной фрейм данных train_test.
train_test['date_str'] = train_test.date.astype(str)
train_test = pd.merge(train_test, oil, how='left', on='date_str')

train_test = pd.merge(train_test, trends, how='left', on='date_str')
train_test.drop(columns='date_str', axis=1, inplace=True)

# Создание новых функций 7-дневного и 28-дневного переходного периода для рекламных акций положительно повлияло на количество заявок.
train_test['onpromo_7'] = train_test['onpromotion'].rolling(7).mean()
train_test['onpromo_28'] = train_test['onpromotion'].rolling(28).mean()
train_test['onpromo_7'].fillna(0, inplace=True)
train_test['onpromo_28'].fillna(0, inplace=True)

# Чтобы добавить данные хранилища в фрейм данных train_test.
stores = pd.read_csv('/kaggle/input/store-sales-time-series-forecasting/stores.csv', index_col='store_nbr')
train_test = pd.merge(train_test, stores, how='left', on='store_nbr')

# Для создания функций национального праздника, включая основную функцию национального праздника, функцию национального мероприятия, функцию национального рабочего дня и определение выходных дней.
holiday = pd.read_csv('/kaggle/input/store-sales-time-series-forecasting/holidays_events.csv')

holiday = holiday.loc[holiday['transferred'] == False]
holiday['description'] = holiday['description'].str.replace('Traslado ', '')

day_off = holiday.where((holiday['type'] != 'Work Day') | (holiday['type'] != 'Event')).set_index('date')['description'].to_dict()
train_test['date_str'] = train_test.date.astype(str)
train_test['national_holiday'] = [1 if a in day_off else 0 for a in train_test.date_str]

event = holiday.where(holiday['type'] == 'Event').set_index('date')['description'].to_dict()
train_test['national_event'] = [1 if a in event else 0 for a in train_test.date_str]

work_day = holiday.where(holiday['type'] == 'Work Day').set_index('date')['description'].to_dict()
train_test['national_workday'] = [1 if a in work_day else 0 for a in train_test.date_str]

train_test['weekend'] = [1 if a >= 5 else 0 for a in train_test.dayofweek]

# Local and Regional
local = holiday.loc[holiday['locale'] == "Local"]
local_dic = local.set_index('date')['locale_name'].to_dict()
train_test['local_holiday'] = [1 if b in local_dic and local_dic[b] == a else 0 for a, b in
                               zip(train_test.city, train_test.date_str)]

regional = holiday.loc[holiday['locale'] == "Regional"]
regional_dic = regional.set_index('date')['locale_name'].to_dict()
train_test['regional_holiday'] = [1 if b in regional_dic and regional_dic[b] == a else 0 for a, b in
                                  zip(train_test.state, train_test.date_str)]

# преобразуйте данные о продажах в журнал, чтобы уменьшить дисперсию и асимметрию.
train_test.sales = np.log1p(train_test.sales)

train_test['Lag_7'] = train_test['sales'].shift(1782 * 7)

train_test['Lag_16'] = train_test['sales'].shift(1782*16)
train_test['Lag_17'] = train_test['sales'].shift(1782*17)
train_test['Lag_18'] = train_test['sales'].shift(1782*18)
train_test['Lag_19'] = train_test['sales'].shift(1782*19)
train_test['Lag_20'] = train_test['sales'].shift(1782*20)
train_test["Lag_21"] = train_test['sales'].shift(1782*21)
train_test['Lag_22'] = train_test['sales'].shift(1782*22)

train_test['Lag_28'] = train_test['sales'].shift(1782 * 28)
train_test['Lag_30'] = train_test['sales'].shift(1782 * 30)
train_test['Lag_31'] = train_test['sales'].shift(1782 * 31)

train_test['Lag_365'] = train_test['sales'].shift(1782 * 365)


def ewm_features(dataframe, alphas, lags):
    for alpha in alphas:
        for lag in lags:
            feature_name = 'ewm_' + str(alpha).replace(".", "") + '_lag_' + str(lag)
            dataframe[feature_name] = dataframe.groupby(["store_nbr", "family"])['sales']. \
                transform(lambda x: x.shift(lag).ewm(alpha=alpha, min_periods=1).mean())
    return dataframe


alphas = [0.95, 0.8, 0.65, 0.5]
lags = [1, 7, 30]
train_test = ewm_features(train_test, alphas, lags)
# Достать все названия столбцов в которых содержится слово lag
#lag_columns = train_test.filter(regex=re.compile(r'(Lag_\d+|oil_lags7_\d+|avg_oil_7|ewm_\d+_\w+_lag_\d+)', re.IGNORECASE)).columns
lags = ['Lag_7','Lag_16','Lag_17','Lag_18','Lag_19','Lag_20','Lag_21','Lag_22','Lag_28', 'Lag_30','Lag_31','Lag_365',
       'oil_lags7_1', 'oil_lags7_2', 'oil_lags7_3', 
        'avg_oil_7', 
        'ewm_095_lag_1', 'ewm_095_lag_7', 'ewm_095_lag_30', 
        'ewm_08_lag_1', 'ewm_08_lag_7', 'ewm_08_lag_30',
        'ewm_065_lag_1', 'ewm_065_lag_7', 'ewm_065_lag_30', 
        'ewm_05_lag_1', 'ewm_05_lag_7', 'ewm_05_lag_30']
train_test[lags] = train_test[lags].fillna(0)

# families = ['AUTOMOTIVE', 'BABY CARE', 'BEAUTY', 'BEVERAGES', 'BOOKS', 'BREAD/BAKERY', 'CELEBRATION', 'CLEANING',
#             'DAIRY', 'DELI', 'EGGS', 'FROZEN FOODS', 'GROCERY I', 'GROCERY II', 'HARDWARE', 'HOME AND KITCHEN I',
#             'HOME AND KITCHEN II', 'HOME APPLIANCES', 'HOME CARE', 'LADIESWEAR', 'LAWN AND GARDEN', 'LINGERIE',
#             'LIQUOR,WINE,BEER', 'MAGAZINES', 'MEATS', 'PERSONAL CARE', 'PET SUPPLIES', 'PLAYERS AND ELECTRONICS',
#             'POULTRY', 'PREPARED FOODS', 'PRODUCE', 'SCHOOL AND OFFICE SUPPLIES', 'SEAFOOD']
families = train_test['family'].unique()
FEATURES = train_test.columns.tolist()
no_features = ('date', 'family', 'sales', 'dayname', 'date_str')

for i in no_features:
    FEATURES.remove(i)

train_test[FEATURES].fillna(0)
TARGET = ['sales']

categories = ['city', 'state', 'type']
for i in categories:
    encoder = preprocessing.LabelEncoder()
    train_test[i] = encoder.fit_transform(train_test[i])

train = train_test[train_test['sales'].notnull()].copy()
test = train_test[train_test['sales'].isnull()].drop(['sales'], axis=1)
# Следующий код выполняет подбор и прогнозирование как для catboost, так и для xgb для каждого семейства продуктов. Поскольку у каждого семейства может быть свое собственное поведение при продажах, этот процесс может обеспечить более точное прогнозирование. Общая оценка с использованием каждого семейства лучше, чем оценка будущих продаж на основе общей глобальной модели.
#
# Кроме того, выполнение прогнозов для тестового набора данных в этом цикле дало лучший результат, чем выполнение его в отдельном цикле.
params = {'lambda': 6.105970537016599, 
          'alpha': 0.874716179324655, 
          'eta': 0.047228549789593455, 
          'colsample_bytree': 0.5, 
          'subsample': 0.7, 
          'learning_rate': 0.012, 
          'n_estimators': 1000, 
          'max_depth': 17, 
          'min_child_weight': 155,
          'early_stopping_rounds': 10
         }
cat_predictions = []
xgb_predictions = []
y_val_cat = pd.DataFrame()
y_val_xgb = pd.DataFrame()
cat = CatBoostRegressor()
xgb = XGBRegressor(**params)

test_predict = pd.DataFrame()

cat_submit = []
xgb_submit = []
print(train_test.select_dtypes(exclude=['int', 'float', 'bool', 'category']).columns)
for family in families:
    train_family = train.loc[train['family'] == family]
    X_train_family,X_val_family,y_train_family,y_val_family = train_test_split(train_family,
                                                                               train_family[TARGET],
                                                                               test_size=0.05,shuffle=False)
    
    cat.fit(X_train_family[FEATURES], y_train_family, eval_set=[(X_train_family[FEATURES],y_train_family),
                                                               (X_val_family[FEATURES], y_val_family)], 
            verbose=False,early_stopping_rounds=10)
    
    xgb.fit(X_train_family[FEATURES], y_train_family, eval_set=[(X_train_family[FEATURES],y_train_family),
                                                                (X_val_family[FEATURES], y_val_family)],
            verbose=False)
    
    cat_pred_family = cat.predict(X_val_family[FEATURES])
    cat_pred_family = [a if a>0 else 0 for a in cat_pred_family]
    cat_predictions.extend(cat_pred_family)
    y_val_cat = pd.concat([y_val_cat, y_val_family])
    
    xgb_pred_family = xgb.predict(X_val_family[FEATURES])
    xgb_pred_family = [a if a>0 else 0 for a in xgb_pred_family]
    xgb_predictions.extend(xgb_pred_family)
    y_val_xgb = pd.concat([y_val_xgb, y_val_family])
    
    test_family = test.loc[test['family'] == family]
    
    cat_pred_submit = cat.predict(test_family[FEATURES])
    cat_pred_submit = [a if a>0 else 0 for a in cat_pred_submit]
    cat_submit.extend(cat_pred_submit)
    
    xgb_pred_submit = xgb.predict(test_family[FEATURES])
    xgb_pred_submit = [a if a>0 else 0 for a in xgb_pred_submit]
    xgb_submit.extend(xgb_pred_submit)
    
    test_predict = pd.concat([test_predict, test_family])
    
    print(family,'CatBoost RMSLE:', np.sqrt(mean_squared_log_error(y_val_family, cat_pred_family)))
    print(family,'XGB RMSLE:', np.sqrt(mean_squared_log_error(y_val_family, xgb_pred_family)))



predictions = [0.5 * a + 0.5 * b for a,b in zip(xgb_submit,cat_submit)] 
test_predict['pred'] = predictions
test_predict.sort_index(inplace=True)

output = pd.read_csv('/kaggle/input/store-sales-time-series-forecasting/sample_submission.csv', index_col='id')
output['sales'] = np.expm1(test_predict['pred'])
output.to_csv('submission_Store_sales.csv')
