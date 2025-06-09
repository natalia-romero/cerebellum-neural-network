import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

from core.regression import CerebellarANN
from cells.cell_types import Granule, Purkinje, DeepNuclei
from core.utils import plot_predictions

SEED = 42
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
data = fetch_california_housing()
X, y = data.data, data.target.reshape(-1, 1)
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
X_train = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
y_train = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)
X_test = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
y_test = torch.tensor(y_test, dtype=torch.float32).to(DEVICE)

# Build model
model = CerebellarANN()
model.add_cell(Granule(plasticity='STDP'))
model.add_cell(Purkinje(plasticity='LTP'))
model.add_cell(DeepNuclei(plasticity='LTD', adaptive=True))
model.to(DEVICE)

# Train model
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
loss_fn = torch.nn.MSELoss()

print("\n📚 Training regression model...")
for epoch in range(100):
    total_loss = 0
    for x, y in zip(X_train, y_train):
        loss = model.train_on_batch(x.unsqueeze(0), y.unsqueeze(0), optimizer, loss_fn)
        total_loss += loss
    if epoch % 10 == 0 or epoch == 99:
        print(f"Epoch {epoch + 1}: Loss = {total_loss / len(X_train):.4f}")

# Evaluate
y_pred = model.predict(X_test)
y_true_inv = scaler_y.inverse_transform(y_test.cpu().numpy().reshape(-1, 1)).flatten()
y_pred_inv = scaler_y.inverse_transform(np.array(y_pred).reshape(-1, 1)).flatten()
r2 = r2_score(y_true_inv, y_pred_inv)

print(f"\nR² Score: {r2:.4f}")
plot_predictions(y_true_inv, y_pred_inv, title=f"Regression Results (R² = {r2:.4f})")
