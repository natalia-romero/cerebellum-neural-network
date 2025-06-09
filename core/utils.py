import matplotlib.pyplot as plt
import numpy as np

def plot_predictions(y_true, y_pred, title='Regression Results'):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.4)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
    plt.xlabel('True')
    plt.ylabel('Predicted')
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

from sklearn.metrics import r2_score

def evaluate_r2(y_true_scaled, y_pred_scaled, scaler_y):
    y_true = scaler_y.inverse_transform(y_true_scaled.reshape(-1, 1)).flatten()
    y_pred = scaler_y.inverse_transform(np.array(y_pred_scaled).reshape(-1, 1)).flatten()
    r2 = r2_score(y_true, y_pred)
    return r2, y_true, y_pred

from sklearn.preprocessing import MinMaxScaler

def scale_data(X, y):
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)
    return X_scaled, y_scaled, scaler_X, scaler_y

import pandas as pd

def generate_dummy_input(features, n_samples=5):
    data = np.random.rand(n_samples, len(features)).astype(np.float32)
    data[:,0] *= 1000  # time_ms
    data[:,1] = (data[:,1] * -30) - 40  # voltage_mV
    data[:,2] *= 0.5  # input_current_nA
    return pd.DataFrame(data, columns=features)

