import pickle
import utils
import pandas as pd

PATH_TEST = "./data/test_df.csv"
df_test = pd.read_csv(PATH_TEST)

with open("model.pkl", "rb") as f:
    clf = pickle.load(f)

X_test, y_test = utils.get_training_samples(df_test)

utils.get_metrics(model=clf, X_test=X_test, y_test=y_test, verbose=False)