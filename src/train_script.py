import pandas as pd
import numpy as np
import pickle

from sklearn.linear_model import LogisticRegression

import utils

PATH_TRAIN = "./data/train_df.csv"
PATH_TEST  = "./data/test_df.csv"
SAVE_MODEL = False


df_train = pd.read_csv(PATH_TRAIN)
df_test  = pd.read_csv(PATH_TEST)

X_train, y_train = utils.get_training_samples(df_train)
X_test, y_test = utils.get_training_samples(df_test)

clf = LogisticRegression(max_iter=2000, class_weight="balanced")
clf.fit(X_train, y_train)

utils.get_metrics(clf, X_test=X_test, y_test=y_test)

### SAVE MODEL
if SAVE_MODEL:
    with open("model.pkl", "wb") as f:
        pickle.dump(clf, f)