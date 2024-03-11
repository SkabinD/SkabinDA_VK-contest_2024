import pickle
import utils
import pandas as pd
import numpy as np

PATH_TEST = "./data/test_df.csv"
df_test = pd.read_csv(PATH_TEST)

with open("model.pkl", "rb") as f:
    clf = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

X_test, y_test = utils.get_training_samples(df_test)
X_test = scaler.transform(X_test)

classes, probabilities, score = utils.get_metrics(model=clf, X_test=X_test, y_test=y_test, 
                                                  verbose=False, return_values=True)

np.savetxt("predicted_labels.csv", classes.astype(int), delimiter=",")