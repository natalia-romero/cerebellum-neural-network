# demo_classifier.py

import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

from core.classifier import CerebellarANNClassifier
from cells.cell_types import Granule, Purkinje, DeepNuclei

SEED = 42
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Simple XOR dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y = np.array([[0], [1], [1], [0]], dtype=np.float32)

scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=SEED)
X_train = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
y_train = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)
X_test = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
y_test = torch.tensor(y_test, dtype=torch.float32).to(DEVICE)

# Build model
model = CerebellarANNClassifier()
model.add_cell(Granule(plasticity='STDP'))
model.add_cell(Purkinje(plasticity='LTP'))
model.add_cell(DeepNuclei(plasticity='LTP'))
model.to(DEVICE)

# Train model
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
loss_fn = torch.nn.BCELoss()

print("\n📚 Training classifier model...")
for epoch in range(200):
    total_loss = 0
    for x, y in zip(X_train, y_train):
        loss = model.train_on_batch(x.unsqueeze(0), y.unsqueeze(0), optimizer, loss_fn)
        total_loss += loss
    if epoch % 20 == 0 or epoch == 199:
        print(f"Epoch {epoch + 1}: Loss = {total_loss / len(X_train):.4f}")

# Evaluate
y_pred = model.predict(X_test)
y_true = y_test.cpu().numpy().flatten()
print("\nClassification Report:")
print(classification_report(y_true, y_pred, digits=4))
