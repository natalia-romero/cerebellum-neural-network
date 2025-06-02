# Reejecutar tras reinicio de entorno
import torch
import torch.nn as nn
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.metrics import r2_score

SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)


class KANFunction(nn.Module):
    def __init__(self, num_basis=12, input_dim=4, output_dim=1):
        super().__init__()
        self.num_basis = num_basis
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.coeffs = nn.Parameter(torch.randn(num_basis * input_dim, output_dim) * 0.5)
        self.knots = torch.linspace(0, 1, steps=num_basis)

    def forward(self, x):
        x_norm = x
        batch_size = x_norm.shape[0]
        expanded_inputs = []
        for k in self.knots:
            expanded_inputs.append(1 - torch.abs(x_norm - k))
        basis_stack = torch.stack(expanded_inputs, dim=-1)
        basis_stack = basis_stack.view(batch_size, -1)
        return torch.matmul(basis_stack, self.coeffs)

class KANNode(nn.Module):
    def __init__(self, num_basis=12, input_dim=4, output_dim=1, plasticity='LTD', inhibition=False, adaptive=False):
        super().__init__()
        self.kan_func = KANFunction(num_basis, input_dim, output_dim)
        self.plasticity = plasticity
        self.inhibition = inhibition
        self.adaptive = adaptive
        self.activity_trace = 0.0
        self.last_input = None
        self.last_output = None

    def forward(self, x):
        self.last_input = x.detach()
        out = self.kan_func(x)

        target_mean_activity = 0.2
        current_mean = out.mean().item()
        self.activity_trace = 0.9 * self.activity_trace + 0.1 * current_mean
        correction = self.activity_trace - target_mean_activity
        out = out - 0.1 * correction

        if self.inhibition:
            inhibition_level = torch.relu(self.last_input).mean().item()
            out = out - 0.1 * inhibition_level

        out = out.view(x.shape[0], -1)
        self.last_output = out.detach()
        return out

    def apply_plasticity(self, error_signal=None):
        with torch.no_grad():
            if self.last_input is None:
                return

            batch_size = self.last_input.shape[0]
            expanded_inputs = []
            for k in self.kan_func.knots:
                expanded_inputs.append(1 - torch.abs(self.last_input - k))
            basis_stack = torch.stack(expanded_inputs, dim=-1)
            basis_stack = basis_stack.view(batch_size, -1)
            expanded_input = basis_stack[0]

            predicted = self.kan_func(self.last_input.unsqueeze(0)).squeeze(0)
            combined_error = error_signal + (predicted - predicted.detach()) if error_signal is not None else predicted - predicted.detach()

            update = 0.01 * (expanded_input.view(-1, 1) @ combined_error.view(1, -1))

            if self.plasticity == 'LTD':
                update = -update
            elif self.plasticity == 'LTP':
                update = +update
            elif self.plasticity == 'STDP':
                update = torch.sign(update) * 0.005

            self.kan_func.coeffs.data.copy_(
                torch.clamp(self.kan_func.coeffs.data + update, min=-1.0, max=1.0)
            )

class CerebellarKAN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cells = nn.ModuleList()

    def add_cell(self, cell):
        self.cells.append(cell)

    def forward(self, x):
        self.intermediate_outputs = []
        for cell in self.cells:
            x = cell(x)
            self.intermediate_outputs.append(x)
        return x

    def apply_plasticity(self, final_error_signal):
        if len(self.cells) > 0:
            self.cells[-1].apply_plasticity(final_error_signal)


def train_regression(model, data, targets, epochs=100):
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    loss_fn = nn.MSELoss()

    train_loss = []

    for epoch in range(epochs):
        total_loss = 0

        for x, y_true in zip(data, targets):
            x = x.unsqueeze(0)
            y_true = y_true.unsqueeze(0)

            y_pred = model(x)
            loss = loss_fn(y_pred, y_true)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            error_signal = (y_true - y_pred).squeeze(0).detach()
            model.apply_plasticity(error_signal)

            total_loss += loss.item()

        avg_loss = total_loss / len(data)
        train_loss.append(avg_loss)

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1}: MSE Loss = {avg_loss:.4f}")

    return train_loss

def evaluate_regression(model, data):
    predictions = []
    with torch.no_grad():
        for x in data:
            x = x.unsqueeze(0)
            y_pred = model(x)
            predictions.append(y_pred.item())

    return predictions


data = fetch_california_housing()
X, y = data.data, data.target.reshape(-1, 1)

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

model = CerebellarKAN()
model.add_cell(KANNode(num_basis=12, input_dim=X_train.shape[1], output_dim=16, plasticity='LTP', inhibition=True))
model.add_cell(KANNode(num_basis=12, input_dim=16, output_dim=1, plasticity='STDP'))


train_loss = train_regression(model, X_train, y_train, epochs=100)
predictions = evaluate_regression(model, X_test)


y_true = scaler_y.inverse_transform(y_test.numpy().reshape(-1, 1)).flatten()
y_pred = scaler_y.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
r2 = r2_score(y_true, y_pred)

plt.figure(figsize=(6, 6))
plt.scatter(y_true, y_pred, alpha=0.4)
plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
plt.xlabel('True')
plt.ylabel('Predicted')
plt.title(f'Regression Results (R2 = {r2:.4f})')
plt.grid(True)
plt.show()
