import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import C
def model_SARIMAX(train_group_data, test_group_data, data_columns):
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
