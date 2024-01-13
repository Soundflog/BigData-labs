import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier as GBR

column_names = [str(i) for i in range(1, 41)]
data = pd.read_csv("train.csv", names=column_names)
target = pd.read_csv("trainLabels.csv", names=["Solution"])

model = GBR()
model.fit(data, target["Solution"].values.ravel())  # Преобразуйте целевую переменную

test_data = pd.read_csv("test.csv", names=column_names)

# Создайте столбец "Идентификатор" от 1 до 9000
identifiers = list(range(1, 9001))
ids = pd.DataFrame(columns=["ID", "Solution"])
ids["ID"] = identifiers
result = model.predict(test_data)
ids["Solution"] = result
ids.to_csv("submission.csv", index=False)
