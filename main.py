import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class KANFunction(nn.Module):
    def __init__(self, num_basis=6):
        super().__init__()
        self.num_basis = num_basis
        self.coeffs = nn.Parameter(torch.randn(num_basis))
        self.knots = torch.linspace(0, 1, steps=num_basis)

    def forward(self, x):
        x_norm = torch.clamp(x, 0, 1)  
        basis = [1 - torch.abs(x_norm - k) for k in self.knots] 
        basis_stack = torch.stack(basis, dim=-1)  
        return (basis_stack * self.coeffs).sum(dim=-1, keepdim=True)


class KANNode(nn.Module):
    def __init__(self, num_basis=6, plasticity=None):
        super().__init__()
        self.kan_func = KANFunction(num_basis)
        self.plasticity = plasticity

    def forward(self, x):
        self.last_input = x.detach()
        return torch.tanh(self.kan_func(x)) 

    def apply_plasticity(self, x, climbing_signal):
        if self.plasticity == 'LTD':
            with torch.no_grad():
                update = -0.01 * (x * climbing_signal).mean()
                self.kan_func.coeffs.data.copy_(self.kan_func.coeffs.data + update)
        elif self.plasticity == 'LTP':
            with torch.no_grad():
                update = +0.01 * (x * (1 - climbing_signal)).mean()
                self.kan_func.coeffs.data.copy_(self.kan_func.coeffs.data + update)


class CerebellarKAN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cells = nn.ModuleList()

    def add_cell(self, cell):
        self.cells.append(cell)

    def forward(self, x, climbing_signal=None):
        for cell in self.cells:
            x = cell(x)
            if hasattr(cell, 'apply_plasticity') and climbing_signal is not None:
                cell.apply_plasticity(x, climbing_signal)
        return x

def train(model, data, targets, epochs=10):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0.0
        for x, y_true in zip(data, targets):
            x = x.unsqueeze(0)
            y_true = y_true.unsqueeze(0)
            climbing_signal = torch.tensor([[random.uniform(0.0, 1.0)]])

            y_pred = model(x, climbing_signal=climbing_signal)
            loss = loss_fn(y_pred, y_true)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")


def predict(model, x):
    with torch.no_grad():
        return model(x.unsqueeze(0))

def evaluate(model, data, targets):
    predictions = []
    errors = []
    for x, y_true in zip(data, targets):
        y_pred = predict(model, x)
        predictions.append(y_pred.item())
        errors.append(abs(y_pred.item() - y_true.item()))
    return predictions, errors


model = CerebellarKAN()
model.add_cell(KANNode(num_basis=6, plasticity=None))       
model.add_cell(KANNode(num_basis=6, plasticity='LTD'))     

data = torch.tensor([[0.0], [0.5], [1.0], [1.5], [2.0]])
targets = torch.tensor([[0.0], [0.25], [0.5], [0.75], [1.0]])

train(model, data, targets, epochs=100)

print("\nEvaluación:")
predictions, errors = evaluate(model, data, targets)
for i, (pred, err) in enumerate(zip(predictions, errors)):
    print(f"Entrada {data[i].item():.2f} Predicción: {pred:.3f} | Error: {err:.3f}")
