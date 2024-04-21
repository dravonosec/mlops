import os
import numpy as np
import pandas as pd


def make_dataset(n_samples, noise=0.1, random_seed=None):
    np.random.seed(random_seed)
    X = np.random.rand(n_samples, 1)
    y = 2 * X + 3 + noise * np.random.randn(n_samples, 1)
    return pd.DataFrame({"X": X.flatten(), "y": y.flatten()})


train_dir = "train"
test_dir = "test"
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

train_set = make_dataset(100, random_seed=42)
test_set = make_dataset(20, random_seed=24)

train_set.to_csv(os.path.join(train_dir, "data.csv"), index=False)
test_set.to_csv(os.path.join(test_dir, "data.csv"), index=False)
