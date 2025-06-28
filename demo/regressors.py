import torch
import numpy as np
import time
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import glob
import os


SEED = 42
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


base_dir = "./demo/A_DeviceMotion_data/A_DeviceMotion_data/"
walk_dirs = sorted(glob.glob(os.path.join(base_dir, "wlk_*")))

X = []
y = []

for walk_dir in walk_dirs:
    for file in sorted(glob.glob(os.path.join(walk_dir, "sub_*.csv"))):
        df = pd.read_csv(file)
        if 'userAcceleration.x' not in df.columns:
            continue
        try:
            features = df.mean(numeric_only=True)
            target = features['userAcceleration.x']
            X.append(features.values)
            y.append(target)
        except Exception as e:
            print(f"File error {file}: {e}")
            continue

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32).reshape(-1, 1)

scaler_X = StandardScaler()
scaler_y = StandardScaler()
X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    start_time = time.time()
    model.fit(X_train, y_train.ravel())
    elapsed = time.time() - start_time
    y_pred = model.predict(X_test).reshape(-1, 1)
    y_test_orig = scaler_y.inverse_transform(y_test)
    y_pred_orig = scaler_y.inverse_transform(y_pred)
    print(f"\n{name} results:")
    print("  Training Time: {} seconds".format(elapsed))
    print("  MSE:", mean_squared_error(y_test_orig, y_pred_orig))
    print("  MAE:", mean_absolute_error(y_test_orig, y_pred_orig))
    print("  R2 :", r2_score(y_test_orig, y_pred_orig))

models = [
    ("Linear Regression", LinearRegression()),
    ("Random Forest", RandomForestRegressor(n_estimators=100, random_state=SEED)),
    ("Support Vector Regressor", SVR()),
    ("MLP Regressor", MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=300, random_state=SEED))
]

for name, model in models:
    evaluate_model(name, model, X_train, y_train, X_test, y_test)