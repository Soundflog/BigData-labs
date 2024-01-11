import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# Перевод секунд в красивое время
def sec_to_time(sec):
    sec = int(sec)
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# Cоединение данных в один датафрейм
def join_data(data, data_oil, data_stores, data_holidays, data_transactions):
    data = pd.merge(data, data_oil, on="date", how="left")
    data = pd.merge(data, data_stores, on="store_nbr", how="left")
    data = pd.merge(data, data_transactions, on=["date", "store_nbr"], how="left")

    data["holiday_type"] = "None"

    for index, row in data_holidays.iterrows():
        date_match = data["date"] == row["date"]
        locale_match = data["state"] == row["locale_name"]
        city_match = data["city"] == row["locale_name"]

        data.loc[date_match & (row["locale"] == "National"), "holiday_type"] = "National"

        data.loc[date_match & locale_match & (row["locale"] == "Regional"), "holiday_type"] = "Regional"

        data.loc[date_match & city_match & (row["locale"] == "Local"), "holiday_type"] = "Local"

        if index % 10 == 0:
            print(f"Обновление {index} из {len(data_holidays)}")

    return data


# Нормализация данных
def get_data(data):
    scaler = MinMaxScaler()
    data_time = pd.to_datetime(data["date"], format="%Y-%m-%d")
    data["day"] = data_time.dt.day
    data["month"] = data_time.index
    data["year"] = data_time.dt.year
    data["day_of_year"] = data_time.dt.dayofyear
    data["day_of_week"] = data_time.dt.dayofweek

    # TODO: Нормализация данных - попробовать убрать
    data["day"] = scaler.fit_transform(data[["day"]])
    data["month"] = scaler.fit_transform(data[["month"]])
    data["year"] = scaler.fit_transform(data[["year"]])
    data["day_of_year"] = scaler.fit_transform(data[["day_of_year"]])
    data["oil_price"] = scaler.fit_transform(data[["oil_price"]])
    data["transactions"] = scaler.fit_transform(data[["transactions"]])

    data = data[["id", "sales", "date", "store_nbr", "family",
                 "holiday_type", "day_of_week",
                 "oil_price", "day", "month", "year", "day_of_year"]]

    data = pd.get_dummies(data, columns=["holiday_type", "day_of_week"], dtype=int)
    data = data.fillna(0)
    return data