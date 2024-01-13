from processing import load_data, submission
from processing_models import apply_model, apply_thread_model, apply_process_model, lr, gbr, sarimax, sarimax_2

if __name__ == '__main__':
    print(f"\n* * * STARTED  * * *")

    # Загрузка и обработка данных
    train_data, text_data, data_columns = load_data(False)

    print(f"\n* * * DATA LOADED  * * *")

    forecast_list = apply_process_model(lr, train_data, text_data, data_columns, 16)
    result = submission(forecast_list, "submission_lab_3_v5_16_lr")

    print(f"\n* * * SUBMISSION  is {result}* * *")

    print(f"\n* * * FINISHED * * *")
