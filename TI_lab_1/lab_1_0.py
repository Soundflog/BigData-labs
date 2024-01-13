import pandas as pd
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

column_names = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11", "C12", "C13", "C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21", "C22", "C23", "C24", "C25", "C26", "C27", "C28", "C29", "C30", "C31", "C32", "C33", "C34", "C35", "C36", "C37", "C38", "C39", "C40"]

scaler = StandardScaler()

# Без нормализации
# 0.87824

# С нормализацией:
# for column in data:
#     data[column] = (data[column] - data[column].min()) / (data[column].max() - data[column].min()) * 2 - 100
# 0.84008

# С нормализацией:
# data = scaler.fit_transform(data)
# data = pd.DataFrame(data, columns=data.columns)
# 0.87792

def normalize(data):
    return data

train_data = pd.read_csv("../python_lab/lab_1/train.csv", names=column_names)
train_data = train_data.loc[:, "C1":"C40"]
train_y = pd.read_csv("../python_lab/lab_1/trainLabels.csv", names=["Solution"])
train_y = train_y.loc[:, "Solution"]


#model = LR()
# 0.80778

#model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=0)
# 0.82853

#model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=0)
# 0.82948

#model = RandomForestClassifier(n_estimators=500, max_depth=10, random_state=0)
# 0.829

#model = RandomForestClassifier(n_estimators=100, max_depth=40, random_state=0)
# 0.82235

#model = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=0)
# 0.82235

#model = DecisionTreeClassifier(max_depth=5, random_state=0)
# 0.79654

#model = DecisionTreeClassifier(max_depth=10, random_state=0)
# 0.78451

#model = KNeighborsClassifier(n_neighbors=5)
# 0.77881

#model = KNeighborsClassifier(n_neighbors=10)
# 0.79069

#model = KNeighborsClassifier(n_neighbors=20)
# 0.80161

#model = KNeighborsClassifier(n_neighbors=40)
# 0.80921

#model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=0)
# 0.83312

#model = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, random_state=0)
# 0.82726

#model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, random_state=0)
# 0.83613

#model = GradientBoostingClassifier(n_estimators=300, learning_rate=0.1, random_state=0)
# 0.83613

#model = GradientBoostingClassifier(n_estimators=400, learning_rate=0.1, random_state=0)
# 0.83787

#model = GradientBoostingClassifier(n_estimators=500, learning_rate=0.1, random_state=0)
# 0.83882

#model = GradientBoostingClassifier(n_estimators=1000, learning_rate=0.1, random_state=0)
# 0.83961
model = GradientBoostingClassifier(n_estimators=2000, learning_rate=0.1, random_state=5634573)
# 0.84008

#model = GradientBoostingClassifier(n_estimators=5000, learning_rate=0.1, random_state=0)
# 0.8385

#model = GradientBoostingClassifier(n_estimators=2000, learning_rate=0.05, random_state=0)
# 0.83818

#model = GradientBoostingClassifier(n_estimators=2000, learning_rate=0.02, random_state=0)
# 0.83581

#model = GradientBoostingClassifier(n_estimators=2000, learning_rate=0.01, random_state=0)
# 0.83296

model.fit(train_data, train_y)

test_data = pd.read_csv("../python_lab/lab_1/test.csv",names=column_names)
test_data = test_data.loc[:, "C1":"C40"]
test_data = normalize(test_data)

id_values = list(range(1, 9001))
ids = pd.DataFrame(columns=["Id", "Solution"])
ids["Id"] = id_values
ids["Solution"] = model.predict(test_data)
ids.to_csv("submission_lab_1.csv", index=False)
