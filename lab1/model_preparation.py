from sklearn.linear_model import LinearRegression
from joblib import dump
import pandas as pd
import os

train_dir = "train"

train_set = pd.read_csv(os.path.join(train_dir, "scaled_data.csv"))

X_train = train_set[["X"]].values
y_train = train_set["y"].values

model = LinearRegression()
model.fit(X_train, y_train)

dump(model, "model.joblib")