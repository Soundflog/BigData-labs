import time
import pandas as pd
from sklearn.svm import SVC as SVM

train_data = pd.read_csv("train.csv")

# # Сдвиг данных
# def shift_data(data, shift_x, shift_y):
#     temp_data = data.copy()
#     data_values = data.values
#     for x in range(28):
#         for y in range(28):
#             shift_coord = (x + shift_x) * 28 + (y + shift_y)
#             if 0 <= shift_coord < temp_data.size:
#                 data_values[x * 28 + y] = temp_data.iloc[shift_coord]
#     return data
#
# def array_shift_data(data, shift_x, shift_y, name="shifted_data"):
#     left_shifted_data = data.copy()
#     start_time = time.time()
#     print("Shift " + name + " :")
#     for i in range(left_shifted_data.shape[0]):
#         left_shifted_data.iloc[i, 1:] = shift_data(left_shifted_data.iloc[i, 1:], shift_x, shift_y)
#         if i % 1000 == 0:
#             print(i / left_shifted_data.shape[0] * 100, "%")
#     end_time = time.time()
#     print("Time: ", end_time - start_time, "\n")
#     return left_shifted_data
#
# def four_size_shift_data(data, space_shift, shift_counter):
#     left_shifted_data = array_shift_data(data, space_shift, 0, f"№{shift_counter}")
#     shift_counter += 1
#     right_shifted_data = array_shift_data(data, -space_shift, 0, f"№{shift_counter}")
#     shift_counter += 1
#     up_shifted_data = array_shift_data(data, 0, -space_shift, f"№{shift_counter}")
#     shift_counter += 1
#     down_shifted_data = array_shift_data(data, 0, space_shift, f"№{shift_counter}")
#     shift_counter += 1
#     data = pd.concat([left_shifted_data, right_shifted_data, up_shifted_data, down_shifted_data], ignore_index=True)
#     return data
#
# shift_counter = 1
# four_size_shift_data(train_data, 3, shift_counter)
# four_size_shift_data(train_data, 6, shift_counter)

# Присвоить первый столбец из train_data в train_y
train_y = train_data["label"]

# Удалить первый столбец из train_data
train_data = train_data.drop("label", axis=1)

# Настройка модели SVM
# model = SVM(verbose=5)
model = SVM(verbose=5, kernel='linear', C=1.0)
# model = SVM(verbose=5, kernel='rbf', C=10.0, gamma=0.1)
# model = SVM(verbose=5, kernel='poly', C=1.0, degree=3)
model.fit(train_data, train_y)

# Тестирование: загрузка тестовых данных и проверка модели
test_data = pd.read_csv("test.csv")
id_values = list(range(1, test_data.shape[0] + 1))
ids = pd.DataFrame(columns=["ImageId", "Label"])
ids["ImageId"] = id_values
ids["Label"] = model.predict(test_data)
ids.to_csv("submission_lab_2_linear.csv", index=False)
# Обычный, Обычный + Сдвиг, linear, rbf, poly
