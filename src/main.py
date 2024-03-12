import pickle
import utils
import pandas as pd
import numpy as np

# Чтение тестовых данных из папки
PATH_TEST = "./data/test_df.csv"
df_test = pd.read_csv(PATH_TEST)

# Загрузка обученных модели и скейлера
with open("model.pkl", "rb") as f:
    clf = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Перевод тестовых данных из pd.DataFrame в np.array и стандартизация
X_test, y_test = utils.get_training_samples(df_test)
X_test = scaler.transform(X_test)

# Предсказание и получение скоров (скор выводится в консоль)
classes, probabilities, score = utils.get_metrics(model=clf, X_test=X_test, y_test=y_test, 
                                                  verbose=False, return_values=True)

# Запись предсказанных классов в файл
np.savetxt("predicted_labels.csv", classes.astype(int), delimiter=",")