import pandas as pd
from sklearn.linear_model import LinearRegression

holidays_events = pd.read_csv("holidays_events.csv")
oil = pd.read_csv("oil.csv")
stores = pd.read_csv("stores.csv")
test = pd.read_csv("test.csv")
train = pd.read_csv("train.csv")
transactions = pd.read_csv("transactions.csv")

oil["dcoilwtico"].bfill(inplace=True)

data = pd.concat([train, test], ignore_index=True)
data = pd.merge(data, oil, on="date", how="left")
data = pd.merge(data, stores, on="store_nbr", how="left")


data["day_of_week"] = pd.to_datetime(data["date"], format="%Y-%m-%d").dt.dayofweek
data = data[["family", "dcoilwtico", "cluster", "day_of_week"]]
data = pd.get_dummies(data, columns=["family", "day_of_week"], dtype=int)
data = data.fillna(0)

model = LinearRegression()
model.fit(data[:train["sales"].shape[0]], train["sales"])

test_y = pd.DataFrame()
test_y["id"] = test["id"]

test = pd.merge(test, oil, on="date", how="left")
test = pd.merge(test, stores, on="store_nbr", how="left")

test["day_of_week"] = pd.to_datetime(test["date"], format="%Y-%m-%d").dt.dayofweek
test = test[["family", "dcoilwtico", "cluster", "day_of_week"]]
test = pd.get_dummies(test, columns=["family", "day_of_week"], dtype=int)
test = test.fillna(0)

test_y["sales"] = model.predict(data[train["sales"].shape[0]:])
test_y.to_csv("submission.csv", index=False)