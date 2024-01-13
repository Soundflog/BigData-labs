import time
import threading
import multiprocessing

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.linear_model import LinearRegression as LR
from statsmodels.tsa.statespace.sarimax import SARIMAX

from processing import sec_to_time


def apply_model(model, train_data_grouped, text_data_grouped, common_data_columns):
    model_count = 0
    forecast_list = list()
    for (store_nbr, family), group in train_data_grouped:
        time_start = time.time()
        model_count += 1

        forecast_temp = model(group, text_data_grouped.get_group((store_nbr, family)), common_data_columns)
        forecast_list.append(forecast_temp)

        print(f"\n*** [{model_count}/{len(train_data_grouped)}] *** "
              f"[{sec_to_time(time.time() - time_start)}] ***")

    return forecast_list


def apply_thread_model(model, train_data_grouped, text_data_grouped, common_data_columns, num_threads=100):
    forecast_list = []
    lock = threading.Lock()  # Для синхронизации доступа к forecast_list

    start_time = time.time()

    def process_group(store_nbr, family, group):
        forecast_temp = model(group, text_data_grouped.get_group((store_nbr, family)), common_data_columns)
        with lock:
            forecast_list.append(forecast_temp)
            print(f"\n*** [{len(forecast_list)}/{len(train_data_grouped)}] *** ")

    threads = []

    for (store_nbr, family), group in train_data_grouped:
        if len(threads) >= num_threads:
            for thread in threads:
                thread.join()
            threads = []
        thread = threading.Thread(target=process_group, args=(store_nbr, family, group))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    print(f"\n TIME: {sec_to_time(time.time() - start_time)}")

    return forecast_list


def worker_function(args):
    store_nbr, family, group, text_data_grouped, common_data_columns, model = args
    result = model(group, text_data_grouped.get_group((store_nbr, family)), common_data_columns)
    return result


def apply_process_model(model, train_data_grouped, text_data_grouped, common_data_columns, num_processes=16):
    time_start = time.time()

    pool = multiprocessing.Pool(processes=num_processes)

    args_list = [(store_nbr, family, group, text_data_grouped, common_data_columns, model) for
                 (store_nbr, family), group in train_data_grouped]

    results = pool.map(worker_function, args_list)

    pool.close()
    pool.join()

    print(f"\n TIME: {sec_to_time(time.time() - time_start)}")

    return results


def sarimax(train_group_data, test_group_data, data_columns):
    train_current_data = train_group_data.copy()

    train_current_data["date"] = pd.to_datetime(train_current_data["date"], format="%Y-%m-%d")
    train_current_data = train_current_data.set_index("date")
    train_current_data = train_current_data.asfreq("D")

    train_y = train_current_data["sales"]
    train_X = train_current_data[data_columns]
    train_X = train_X.fillna(0)

    order = (1, 1, 1)
    seasonal_order = (1, 1, 1, 12)
    model = SARIMAX(train_y, order=order, seasonal_order=seasonal_order, exog=train_X)
    model_fit = model.fit(maxiter=1000, method='powell', disp=False)

    test_current_data = test_group_data.copy()

    test_X = test_current_data[data_columns]
    test_X = test_X.fillna(0)

    forecast = model_fit.get_forecast(steps=len(test_current_data), exog=test_X)

    forecast_values = forecast.predicted_mean

    forecast_group = pd.DataFrame({"id": test_current_data["id"]})
    forecast_group["sales"] = forecast_values.values
    forecast_group = forecast_group.dropna(axis=1, how='all')

    return forecast_group


def sarimax_2(train_group_data, test_group_data, data_columns):
    train_current_data = train_group_data.copy()

    train_current_data["date"] = pd.to_datetime(train_current_data["date"], format="%Y-%m-%d")
    train_current_data = train_current_data.set_index("date")
    train_current_data = train_current_data.asfreq("D")

    train_y = train_current_data["sales"]
    train_X = train_current_data[data_columns]
    train_X = train_X.fillna(0)

    order = (2, 1, 2)
    seasonal_order = (2, 3, 5, 12)
    model = SARIMAX(train_y, order=order, seasonal_order=seasonal_order, exog=train_X)
    model_fit = model.fit(maxiter=1000, method='powell', disp=False)

    test_current_data = test_group_data.copy()

    test_X = test_current_data[data_columns]
    test_X = test_X.fillna(0)

    forecast = model_fit.get_forecast(steps=len(test_current_data), exog=test_X)

    forecast_values = forecast.predicted_mean

    forecast_group = pd.DataFrame({"id": test_current_data["id"]})
    forecast_group["sales"] = forecast_values.values
    forecast_group = forecast_group.dropna(axis=1, how='all')

    return forecast_group


# def prophet_forecast(train_group_data, test_group_data, data_columns):
#     train_current_data = train_group_data.copy()
#
#     train_current_data["date"] = pd.to_datetime(train_current_data["date"], format="%Y-%m-%d")
#     train_current_data = train_current_data.rename(columns={"date": "ds", "sales": "y"})
#
#     model = Prophet()
#
#     for col in data_columns:
#         if col != "ds" and col != "y":
#             model.add_regressor(col)
#
#     model.fit(train_current_data)
#
#     test_current_data = test_group_data.copy()
#     test_current_data["date"] = pd.to_datetime(test_current_data["date"], format="%Y-%m-%d")
#     test_current_data = test_current_data.rename(columns={"date": "ds"})
#
#     forecast = model.predict(test_current_data)
#
#     forecast_group = pd.DataFrame({"id": test_current_data["id"]})
#     forecast_group["sales"] = forecast["yhat"].values
#     forecast_group = forecast_group.dropna(axis=1, how='all')
#
#     return forecast_group


def lr(train_group_data, test_group_data, data_columns):
    train_current_data = train_group_data.copy()

    train_X = train_current_data[data_columns]
    train_X = train_X.fillna(0)

    train_y = train_current_data["sales"]

    model = LR()

    train_X.to_csv("train_X.csv", index=False)
    model.fit(train_X, train_y)

    test_current_data = test_group_data.copy()

    test_X = test_current_data[data_columns]
    test_X = test_X.fillna(0)

    test_y = pd.DataFrame()
    test_y["id"] = test_current_data["id"]
    test_y["sales"] = model.predict(test_X)

    return test_y


def gbr(train_group_data, test_group_data, data_columns):
    train_current_data = train_group_data.copy()

    train_X = train_current_data[data_columns]
    train_X = train_X.fillna(0)

    train_y = train_current_data["sales"]

    model = GBR(n_estimators=250, max_depth=5, min_samples_split=2, learning_rate=0.1)
    model.fit(train_X, train_y)

    test_current_data = test_group_data.copy()

    test_X = test_current_data[data_columns]
    test_X = test_X.fillna(0)

    test_y = pd.DataFrame()
    test_y["id"] = test_current_data["id"]
    test_y["sales"] = model.predict(test_X)

    return test_y
