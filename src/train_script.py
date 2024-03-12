import pandas as pd
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

import utils

# Путь к данным
PATH_TRAIN = "./data/train_df.csv"
PATH_TEST  = "./data/test_df.csv"
# Сохранение модели в конце обучения
SAVE_MODEL = False

# Подготовка данных для обучения
df_train = pd.read_csv(PATH_TRAIN)
df_test  = pd.read_csv(PATH_TEST)

X_train, y_train = utils.get_training_samples(df_train, synthetic=False)
X_test, y_test = utils.get_training_samples(df_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Обучение и вывод метрик
clf = LogisticRegression(max_iter=2000, class_weight="balanced")
clf.fit(X_train_scaled, y_train)

utils.get_metrics(clf, X_test=X_test_scaled, y_test=y_test)

### SAVE MODEL
if SAVE_MODEL:
    with open("model.pkl", "wb") as f:
        pickle.dump(clf, f)
    
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)