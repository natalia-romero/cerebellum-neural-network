import torch
import torch.nn as nn

class CerebellarANNClassifier(nn.Module):
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

    def apply_plasticity(self, error_signal):
        if self.cells:
            self.cells[-1].apply_plasticity(error_signal)

    def train_on_batch(self, x, y_true, optimizer, loss_fn):
        y_pred = torch.sigmoid(self.forward(x))
        loss = loss_fn(y_pred, y_true)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        error_signal = (y_true - y_pred).squeeze(0).detach()
        self.apply_plasticity(error_signal)
        return loss.item()

    def predict_proba(self, X):
        self.eval()
        probabilities = []
        with torch.no_grad():
            for x in X:
                x = x.unsqueeze(0)
                y_pred = torch.sigmoid(self.forward(x)).item()
                probabilities.append(y_pred)
        return probabilities

    def predict(self, X, threshold=0.5):
        probabilities = self.predict_proba(X)
        return [int(p > threshold) for p in probabilities]
