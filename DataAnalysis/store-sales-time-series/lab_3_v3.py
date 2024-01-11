import time

import pandas as pd

import processing
import processing_models as models

# Загрузка данных

holidays = pd.read_csv("holidays_events.csv")
print("* Holiday data loaded")

stores = pd.read_csv("stores.csv")
print("* Stores data loaded")

oil = pd.read_csv("oil.csv")
print("* Oil data loaded")

transactions = pd.read_csv("transactions.csv")
print("* Transactions data loaded")

train_data = pd.read_csv("train.csv")
print("* Train data loaded")

test_data = pd.read_csv("test.csv")
print("* Test data loaded")

common_data = pd.concat([train_data, test_data], ignore_index=True)

# Обработка метаданных

oil["dcoilwtico"].bfill(inplace=True)

oil.rename(columns={"dcoilwtico": "oil_price"}, inplace=True)

holidays = holidays[~holidays["transferred"] & ((holidays["type"] != "Work Day") | (holidays["type"] != "Event"))]

holidays = holidays.sort_values(by=["date", "locale"], ascending=[True, True])

# TODO Подготовка данных
common_data = processing.join_data(common_data, oil, stores, holidays, transactions)
# common_data.to_csv("common_data.csv", index=False)

# TODO УСКОРЕННАЯ ЗАГРУЗКА ДАННЫХ ДЛЯ ТЕСТИРОВАНИЯ ПОСЛЕ ФУНКЦИИ "join_data"
# common_data = pd.read_csv("common_data.csv")
# common_data["transactions"] = common_data["transactions"].fillna(common_data["transactions"].mean())

common_data = processing.get_data(common_data)

common_data_columns = common_data.columns[5:]

train_data_grouped = (common_data.iloc[:train_data.shape[0]].groupby(["store_nbr", "family"]))

text_data_grouped = (common_data.iloc[train_data.shape[0]:].groupby(["store_nbr", "family"]))

model_count = 0

# Фрейм для предсказаний
forecast_columns = ["id", "sales"]
forecast_df = pd.DataFrame(columns=forecast_columns)

forecast_list = list()
print(f"\n* * * STARTED  * * *")
for (store_nbr, family), group in train_data_grouped:
    time_start = time.time()
    model_count += 1
    forecast_temp = models.model_SARIMAX(group, text_data_grouped.get_group((store_nbr, family)), common_data_columns)
    forecast_list.append(forecast_temp)
    print(
        f"\n* * * [{model_count}/{len(train_data_grouped)}] * * * [{processing.sec_to_time(time.time() - time_start)}] * * *")

forecast_df = pd.concat(forecast_list, axis=0, ignore_index=True)
forecast_df = forecast_df.sort_values(by=["id"], ascending=[True])
forecast_df.to_csv("subs/submission_lab_3.csv", index=False)
