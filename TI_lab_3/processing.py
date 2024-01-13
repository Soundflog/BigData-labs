import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# Перевод секунд в красивое время
def sec_to_time(sec):
    sec = int(sec)
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# Обработка метаданных
def processing_metadata(oil, holidays):
    oil["dcoilwtico"].bfill(inplace=True)
    oil.rename(columns={"dcoilwtico": "oil_price"}, inplace=True)

    holidays["holiday_type"] = ""
    holidays.loc[(holidays["type"] == "Holiday") | (holidays["type"] == "Additional") |
                 (holidays["type"] == "Bridge") | (holidays["type"] == "Transfer"), "holiday_type"] = "Holiday"
    holidays.loc[holidays["type"] == "Work Day", "holiday_type"] = "Work Day"
    holidays.loc[holidays["type"] == "Event", "holiday_type"] = "Event"
    holidays = holidays[~holidays["transferred"]]

    return oil, holidays


# Загрузка данных
def load_data(is_load_data=True):
    holidays = pd.read_csv("holidays_events.csv")
    stores = pd.read_csv("stores.csv")
    oil = pd.read_csv("oil.csv")
    transactions = pd.read_csv("transactions.csv")
    train_data = pd.read_csv("train.csv")
    test_data = pd.read_csv("test.csv")
    common_data = pd.concat([train_data, test_data], ignore_index=True)
    oil, holidays = processing_metadata(oil, holidays)

    if is_load_data:
        common_data = join_data(common_data, oil, stores, holidays, transactions)
        common_data.to_csv("common_data.csv", index=False)
    else:
        common_data = pd.read_csv("common_data.csv")
        common_data["transactions"] = common_data["transactions"].fillna(common_data["transactions"].mean())

    common_data = get_data(common_data)

    common_data_columns = common_data.columns[5:]
    train_data_grouped = common_data.iloc[:train_data.shape[0]].groupby(["store_nbr", "family"])
    text_data_grouped = common_data.iloc[train_data.shape[0]:].groupby(["store_nbr", "family"])

    return train_data_grouped, text_data_grouped, common_data_columns


# Cоединение данных в один датафрейм
def join_data(data, data_oil, data_stores, data_holidays, data_transactions):
    # Слияние данных
    data = pd.merge(data, data_oil, on="date", how="left")
    data = pd.merge(data, data_stores, on="store_nbr", how="left")
    data = pd.merge(data, data_transactions, on=["date", "store_nbr"], how="left")

    # Создание новых столбцов
    data["holiday_type_National"] = 0
    data["holiday_type_Regional"] = 0
    data["holiday_type_Local"] = 0
    data["work_day"] = 0
    data["event"] = 0
    data["earthquake"] = 0
    data["christmas"] = 0
    data["football"] = 0
    data["black_friday"] = 0
    data["cyber_monday"] = 0
    data["carnival"] = 0

    # Создание словаря, в котором ключами будут даты, а значениями списки строк
    holiday_dates = {}
    for _, row in data_holidays.iterrows():
        date = row["date"]
        if date not in holiday_dates:
            holiday_dates[date] = []
        holiday_dates[date].append(row)

    # Обновление новых столбцов на основе данных о праздниках
    for i, row in data.iterrows():
        date = row["date"]
        if date in holiday_dates:
            for holiday in holiday_dates[date]:
                if holiday["holiday_type"] == "Holiday" and holiday["locale"] == "National":
                    data.at[i, "holiday_type_National"] = 1

                if (holiday["holiday_type"] == "Holiday" and holiday["locale"] == "Regional" and
                        row["state"] == holiday["locale_name"]):
                    data.at[i, "holiday_type_Regional"] = 1

                if (holiday["holiday_type"] == "Holiday" and holiday["locale"] == "Local" and
                        row["city"] == holiday["locale_name"]):
                    data.at[i, "holiday_type_Local"] = 1

                if holiday["holiday_type"] == "Work Day":
                    data.at[i, "work_day"] = 1

                if holiday["holiday_type"] == "Event":
                    data.at[i, "event"] = 1

                if "Terremoto Manabi" in holiday["description"]:
                    data.at[i, "earthquake"] = 1

                if "Navidad" in holiday["description"]:
                    data.at[i, "christmas"] = 1

                if "futbol" in holiday["description"]:
                    data.at[i, "football"] = 1

                if "Black Friday" in holiday["description"]:
                    data.at[i, "black_friday"] = 1

                if "Cyber Monday" in holiday["description"]:
                    data.at[i, "cyber_monday"] = 1

                if "Carnaval" in holiday["description"]:
                    data.at[i, "carnival"] = 1

        if i % 300000 == 0:
            print(f"Process: [{i}/{len(data)}] [{(round(i / len(data) * 100, 1))} %]")

    return data


# Нормализация данных
def get_data(data):
    data_time = pd.to_datetime(data["date"], format="%Y-%m-%d")
    data["day"] = data_time.dt.day
    data["month"] = data_time.dt.month
    data["year"] = data_time.dt.year
    data["day_of_year"] = data_time.dt.dayofyear
    data["day_of_week"] = data_time.dt.dayofweek

    data["oil_price"] = data["oil_price"].interpolate().round(2)

    scaler = MinMaxScaler()
    data["day"] = scaler.fit_transform(data[["day"]])
    data["month"] = scaler.fit_transform(data[["month"]])
    data["year"] = scaler.fit_transform(data[["year"]])
    data["day_of_year"] = scaler.fit_transform(data[["day_of_year"]])
    data["oil_price"] = scaler.fit_transform(data[["oil_price"]])
    data["transactions"] = scaler.fit_transform(data[["transactions"]])

    data = data[["id", "sales", "date", "store_nbr", "family",
                 "day_of_week",
                 "oil_price", "day", "month", "year", "day_of_year",
                 "holiday_type_National", "holiday_type_Regional", "holiday_type_Local", "work_day", "event",
                 "earthquake", "christmas", "football", "black_friday", "cyber_monday", "carnival"]]

    data = pd.get_dummies(data, columns=["day_of_week"], dtype=int)
    data = data.fillna(0)
    return data


# Вывод результатов
def submission(forecast_list, title="submission"):
    pd.DataFrame(columns=["id", "sales"])
    forecast_df = pd.concat(forecast_list, axis=0, ignore_index=True)
    forecast_df = forecast_df.sort_values(by=["id"], ascending=[True])
    forecast_df.to_csv(f"{title}.csv", index=False)
    return "OK"
