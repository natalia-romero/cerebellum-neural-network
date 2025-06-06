import torch
import torch.nn as nn
import random

SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)

class KANFunction(nn.Module):
    def __init__(self, num_basis=6):
        super().__init__()
        self.num_basis = num_basis
        self.coeffs = nn.Parameter(torch.randn(num_basis) * 0.5)
        self.knots = torch.linspace(0, 1, steps=num_basis)

    def forward(self, x):
        x_norm = x  # Asumimos que x ya está normalizado
        basis = [1 - torch.abs(x_norm - k) for k in self.knots]
        basis_stack = torch.stack(basis, dim=-1)
        return (basis_stack * self.coeffs).sum(dim=-1, keepdim=True)

class KANNode(nn.Module):
    def __init__(self, num_basis=6, plasticity='LTD', inhibition=False, adaptive=False):
        super().__init__()
        self.kan_func = KANFunction(num_basis)
        self.plasticity = plasticity
        self.inhibition = inhibition
        self.adaptive = adaptive
        self.activity_trace = 0.0

    def forward(self, x):
        out = self.kan_func(x)

        if self.inhibition:
            out = out - 0.1 * self.activity_trace

        if self.adaptive:
            out = out * (1.0 - 0.05 * self.activity_trace)

        out = torch.tanh(out)
        out = out.view(x.shape[0], -1)
        self.activity_trace = 0.9 * self.activity_trace + 0.1 * out.mean().item()
        return out

    def apply_plasticity(self, error_signal):
        with torch.no_grad():
            if self.plasticity == 'LTD':
                update = -0.01 * error_signal.mean(dim=0)
            elif self.plasticity == 'LTP':
                update = +0.01 * error_signal.mean(dim=0)
            elif self.plasticity == 'STDP':
                update = torch.sign(error_signal).mean(dim=0) * 0.005
            else:
                return
            update = update.squeeze()
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
        for cell in self.cells:
            x = cell(x)
        return x.view(x.shape[0], -1)

    def apply_plasticity(self, error_signal):
        for cell in self.cells:
            if hasattr(cell, 'apply_plasticity'):
                cell.apply_plasticity(error_signal)

def train(model, data, targets, epochs=50):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.01)
    loss_fn = nn.SmoothL1Loss()

    for epoch in range(epochs):
        total_loss = 0.0
        for i, (x, y_true) in enumerate(zip(data, targets)):
            x = x.unsqueeze(0)
            y_true = y_true.unsqueeze(0)

            y_pred = model(x)
            error_signal = (y_true - y_pred).detach()

            # aplicar plasticidad basada en el error
            model.apply_plasticity(error_signal)

            loss = loss_fn(y_pred, y_true)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")
        with torch.no_grad():
            out = model(data)
            print(f"  Outputs: {[round(val.item(), 3) for val in out.view(-1)]}")

def predict(model, x):
    with torch.no_grad():
        return model(x.unsqueeze(0))

def evaluate(model, data, targets):
    predictions = []
    errors = []
    squared_errors = []
    total_variance = 0.0
    mean_target = targets.mean().item()

    for x, y_true in zip(data, targets):
        y_pred = predict(model, x)
        predictions.append(y_pred.item())
        error = abs(y_pred.item() - y_true.item())
        squared_error = (y_pred.item() - y_true.item()) ** 2

        errors.append(error)
        squared_errors.append(squared_error)
        total_variance += (y_true.item() - mean_target) ** 2

    mse = sum(squared_errors) / len(squared_errors)
    mae = sum(errors) / len(errors)
    r2 = 1 - (sum(squared_errors) / total_variance)

    print(f"\nMetrics:")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R² Score: {r2:.4f}\n")

    return predictions, errors

model = CerebellarKAN()
model.add_cell(KANNode(num_basis=6, plasticity='LTP'))
model.add_cell(KANNode(num_basis=6, plasticity='LTP'))
model.add_cell(KANNode(num_basis=12, plasticity='STDP', inhibition=True, adaptive=True))
model.add_cell(KANNode(num_basis=12, plasticity='LTP'))

model.add_cell(KANNode(num_basis=6, plasticity='LTP'))
data = torch.tensor([[0.0], [0.5], [1.0], [1.5], [2.0]])
data = (data - data.min()) / (data.max() - data.min()) 
targets = torch.tensor([[0.0], [0.25], [0.5], [0.75], [1.0]])

train(model, data, targets, epochs=25)

print("\nEvaluación:")
predictions, errors = evaluate(model, data, targets)

for i, (pred, err) in enumerate(zip(predictions, errors)):
    print(f"Entrada {data[i].item():.2f} Predicción: {pred:.3f} | Error: {err:.3f}")
