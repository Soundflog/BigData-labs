import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.linear_model import LinearRegression as LR
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

train_data = pd.read_csv('train.csv', parse_dates=['date'])

holidays_events_data = pd.read_csv('holidays_events.csv', parse_dates=['date'])
# print(f"Holidays events: \n {holidays_events_data.head()}")


oil_data = pd.read_csv('oil.csv', parse_dates=['date'])
# print(f"Oil: \n{oil_data.head()}")
oil_data['dcoilwtico'] = oil_data['dcoilwtico'].interpolate().round(2)
oil_data.loc[0, 'dcoilwtico'] = oil_data.loc[1, 'dcoilwtico']

stores_data = pd.read_csv('stores.csv')
# print(f"Stores data: \n {stores_data.head()}")

transactions_data = pd.read_csv('transactions.csv', parse_dates=['date'])
test_data = pd.read_csv('test.csv', parse_dates=['date'])


# print(f"Transactions: \n{transactions_data.head()}")


# Merge
def merge_data(data):
    data = pd.merge(data, stores_data, on="store_nbr", how="left")
    data = pd.merge(data, oil_data, on="date", how="left")
    data = pd.merge(data, holidays_events_data, on='date', how='left')
    return data


def preprocess_data(df):
    holiday_store_list = []

    for i in range(len(stores_data)):
        df_holiday_dummies = pd.DataFrame(columns=['date'])
        df_holiday_dummies["date"] = df["date"]
        df_holiday_dummies["store_nbr"] = i + 1

        df_holiday_dummies["national_holiday"] = np.where(
            ((df["type"] == "Holiday") & (df["locale"] == "National")), 1, 0)

        df_holiday_dummies["earthquake_relief"] = np.where(
            df['description'].str.contains('Terremoto Manabi'), 1, 0)

        df_holiday_dummies["christmas"] = np.where(df['description'].str.contains('Navidad'), 1, 0)

        df_holiday_dummies["football_event"] = np.where(df['description'].str.contains('futbol'), 1, 0)

        df_holiday_dummies["national_event"] = np.where(((df["type"] == "Event") & (
                df["locale"] == "National") & (~df['description'].str.contains('Terremoto Manabi'))
                                                         & (~df['description'].str.contains('futbol'))), 1, 0)

        df_holiday_dummies["work_day"] = np.where((df["type"] == "Work Day"), 1, 0)

        df_holiday_dummies["local_holiday"] = np.where(((df["type"] == "Holiday") & (
                (df["locale_name"] == stores_data['state'][i]) | (
                df["locale_name"] == stores_data['city'][i]))), 1, 0)

        df_holiday_dummies = df_holiday_dummies[~df_holiday_dummies['date'].duplicated(keep='first')]

        holiday_store_list.append(df_holiday_dummies)

    holiday_store_df = pd.concat(holiday_store_list)

    future_merged = pd.merge(train_data_concat, holiday_store_df, on=['date', 'store_nbr'], how='left')
    future_merged = pd.merge(future_merged, oil_data, on="date", how="left")
    # future_merged = pd.merge(future_merged, transactions_data, on=['date', 'store_nbr'], how='left')

    future_merged['dcoilwtico'] = future_merged['dcoilwtico'].interpolate().round(2)
    return future_merged


def process_date_columns(data):
    data = data.set_index(['date'])
    data['day'] = data.index.day
    data['dayofweek'] = data.index.dayofweek
    data['dayofyear'] = data.index.dayofyear
    data['month'] = data.index.month
    data['year'] = data.index.year

    dow_ohe = pd.get_dummies(data['dayofweek'])
    dow_ohe = dow_ohe.drop([0], axis=1)

    data = data.drop(['dayofweek'], axis=1)
    data[dow_ohe.columns] = dow_ohe

    # scaled_cols = ['dcoilwtico', 'onpromotion', 'day', 'dayofyear', 'month', 'year']
    # data.columns = data.columns.astype(str)
    # data[scaled_cols] = scaler.fit_transform(data[scaled_cols])
    #
    # data['transactions'] = scaler.fit_transform(data[['transactions']])
    return data


train_data_concat = pd.concat([train_data, test_data], ignore_index=True)
train_df = preprocess_data(holidays_events_data)
# train_df = process_date_columns(train_df)

# Создаем словарь для соответствия категорий праздников
# holiday_mapping = {'NaN': 0, 'National': 3, 'Regional': 2, 'Local': 1}
# df_grouped = holidays_events_data.groupby('date')['locale'].apply(lambda x: x.map(holiday_mapping).max())
# train_data['holiday_category'] = train_data['date'].map(df_grouped)
# train_data['locale'] = train_data['locale'].map(holiday_mapping)

# train_data.to_csv('data_train_test.csv')
# model = GBC(random_state=544_433)
model = LR()
del train_df['family']
# train_df = pd.get_dummies(train_df, columns=['family'], dtype=int)

# national_holiday,earthquake_relief,Christmas,football_event,national_event,work_day,local_holiday
train_df['national_holiday'].fillna(0.0, inplace=True)
train_df['earthquake_relief'].fillna(0.0, inplace=True)
train_df['christmas'].fillna(0.0, inplace=True)
train_df['football_event'].fillna(0.0, inplace=True)
train_df['national_event'].fillna(0.0, inplace=True)
train_df['work_day'].fillna(0.0, inplace=True)
train_df['local_holiday'].fillna(0.0, inplace=True)

print(train_df.head())

data_train = train_df.iloc[:train_data.shape[0]]
data_train.to_csv('process_csv/data_train_test.csv')
print(f"TRAIN:\n {data_train.head()}")

test_data = train_df.iloc[train_data.shape[0]:]
test_data.to_csv('process_csv/test_data.csv')

y = data_train['sales']
X = data_train.drop(columns=['sales'])

model.fit(X, y)

predictions_data = model.predict(test_data.drop(columns=['sales'])).round(2)
# присвоить 0.0 отрицательным значениям массива
predictions_data = np.where(predictions_data < 0.0, 0.0, predictions_data)
# id,sales
sample_submission = pd.read_csv('sample_submission.csv')
submission_df = pd.DataFrame({'id': sample_submission['id'], 'sales': predictions_data})

submission_df.to_csv("subs/submission.csv", index=False)
# print(f"Test data: \n {test_data.head()}")
