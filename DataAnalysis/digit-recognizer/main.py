import pandas as pd
from sklearn.svm import SVC

data_train = pd.read_csv('train.csv')
data_test = pd.read_csv('test.csv')

data_train_X = data_train.drop(columns='label')
data_train_Y = data_train['label']

# from sklearn.preprocessing import StandardScaler
# # Нормализация данных
# scaler = StandardScaler()
# data_train_X = scaler.fit_transform(data_train_X)

print(f"data train : \n {data_train.head()}")
# model = GBC(n_estimators=600, random_state=644_323)
# 0.96414
# model = KNC(n_neighbors=10)
# model = KNC(n_neighbors=5)
# 0.97775
# model = SVC(C=1.2, kernel='poly', gamma='auto', degree=2, coef0=0.8, random_state=25_555)
# 0.97825
model = SVC(kernel='poly', gamma='scale', coef0=0.5, random_state=25_555)
model.fit(data_train_X, data_train_Y)

id_values = list(range(1, data_test.shape[0] + 1))
ids = pd.DataFrame(columns=["ImageId", "Label"])
ids["ImageId"] = id_values
ids["Label"] = model.predict(data_test)
ids.to_csv("submission.csv", index=False)


