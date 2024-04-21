from joblib import load
from sklearn.metrics import mean_squared_error
import pandas as pd
import os

test_dir = "test"

test_set = pd.read_csv(os.path.join(test_dir, "scaled_data.csv"))

X_test = test_set[["X"]].values
y_test = test_set["y"].values

model = load("model.joblib")

predictions = model.predict(X_test)

mse = mean_squared_error(y_test, predictions)

print("Model Apply MSE: {}".format(mse))