import pandas as pd
from sklearn.neural_network import MLPClassifier as MLPC


train_data = pd.read_csv("train.csv", header=None)
train_labels = pd.read_csv("trainLabels.csv", header=None)
test_data = pd.read_csv("test.csv", header=None)


model = MLPC(random_state=63_716_841)

# .values.ravel() для преобразования меток в одномерный массив
model.fit(train_data, train_labels.values.ravel())

# Предсказание классов для тестового набора данных
test_predictions = model.predict(test_data)

submission_df = pd.DataFrame({'Id': range(1, len(test_predictions) + 1), 'Solution': test_predictions})

submission_df.to_csv("submissionTest.csv", index=False)

