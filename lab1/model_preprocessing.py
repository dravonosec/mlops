from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import os

train_dir = "train"
test_dir = "test"

train_set = pd.read_csv(os.path.join(train_dir, "data.csv"))
test_set = pd.read_csv(os.path.join(test_dir, "data.csv"))

scaler = MinMaxScaler()
scaled_train_set = pd.DataFrame(scaler.fit_transform(train_set), columns=train_set.columns)
scaled_test_set = pd.DataFrame(scaler.transform(test_set), columns=test_set.columns)

scaled_train_set.to_csv(os.path.join(train_dir, "scaled_data.csv"), index=False)
scaled_test_set.to_csv(os.path.join(test_dir, "scaled_data.csv"), index=False)