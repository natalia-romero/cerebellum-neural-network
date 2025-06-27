import torch
import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

from core.regressor import CerebellarANNRegressor
from cells.cell_types import Granule, Purkinje, DeepNuclei, Basket, MossyFiber, ClimbingFiber, Stellate

SEED = 42
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#data
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
            print(f"⚠️ Error con archivo {file}: {e}")
            continue

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32).reshape(-1, 1)

scaler_X = StandardScaler()
scaler_y = StandardScaler()
X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=SEED)
X_train = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
y_train = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)
X_test = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
y_test = torch.tensor(y_test, dtype=torch.float32).to(DEVICE)

#model
model = CerebellarANNRegressor()
model.add_cell(Purkinje(plasticity='LTD'))  
model.finalize()
model.to(DEVICE)

params = []
for cell in model.cells:
    for p in cell.model.parameters():
        if p.requires_grad:
            params.append(p)

optimizer = torch.optim.AdamW(params, lr=0.01)
loss_fn = torch.nn.MSELoss()

#train
print("\nTraining regressor model...")
for epoch in range(1, 101):
    loss = model.train_on_batch(X_train, y_train, optimizer, loss_fn)
    if epoch % 10 == 1 or epoch == 100:
        print(f"Epoch {epoch}: Loss = {loss:.4f}")

#evaluation
model.eval()
with torch.no_grad():
    y_pred = model.predict(X_test)
    y_pred_inv = scaler_y.inverse_transform(y_pred.reshape(-1, 1))
    y_test_inv = scaler_y.inverse_transform(y_test.cpu().numpy().reshape(-1, 1))

mse = mean_squared_error(y_test_inv, y_pred_inv)
r2 = r2_score(y_test_inv, y_pred_inv)
print("\nEvaluation:")
print("MSE:", round(mse, 4))
print("R² :", round(r2, 4))
plt.figure(figsize=(10, 6))
plt.plot(y_test_inv, label="Ground Truth", marker='o')
plt.plot(y_pred_inv, label="Prediction", marker='x')
plt.title("CINN Regression - MotionSense Dataset")
plt.xlabel("Sample Index")
plt.ylabel("Target Value (e.g., mean acc x-axis)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("motion_regression_plot.png")  