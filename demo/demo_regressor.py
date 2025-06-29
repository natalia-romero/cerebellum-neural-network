import torch
import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from core.regressor import CINNRegressor
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
            print(f"File error {file}: {e}")
            continue

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32).reshape(-1, 1)

scaler_X = StandardScaler()
scaler_y = StandardScaler()
X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
X_train = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
y_train = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)
X_test = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
y_test = torch.tensor(y_test, dtype=torch.float32).to(DEVICE)

# Build model
model = CINNRegressor()
model.add_cell(Granule(plasticity='STDP', inhibition=True))
model.add_cell(Purkinje(plasticity='LTP', inhibition=False))
model.add_cell(DeepNuclei(plasticity='LTD', inhibition=True))
model.finalize()
model.to(DEVICE)

optimizer = torch.optim.AdamW(model.get_trainable_parameters(), lr=0.001)
loss_fn = torch.nn.MSELoss()

# Train
print("\nTraining model...")
start_time = time.time()
losses = model.fit(X_train, y_train, optimizer, loss_fn, epochs=100)
training_time = time.time() - start_time
print(f"Training time: {training_time:.2f} seconds")

plt.figure()
plt.plot(range(1, len(losses)+1), losses, marker='o')
plt.title("Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.tight_layout()
os.makedirs("results_regression", exist_ok=True)
plt.savefig("results_regression/loss_curve.png")
plt.close()


#eva
model.evaluate(X_test, y_test, scaler_y=scaler_y, return_predictions=True)

print(f"Training time: {training_time} seconds\n")
