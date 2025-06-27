import torch
import torch.nn as nn

class CerebellarANNClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.cells = []
        self.linear = None  # Se define en finalize()

    def add_cell(self, cell):
        self.cells.append(cell)

    def finalize(self):
        # Pasar dummy input por la red para inicializar last_output
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.cells[0].input_dim)
            out = self.forward_internal(dummy_input)
        
        output_dim = out.shape[1]
        print("✅ Red finalizada. output_dim =", output_dim)
        self.linear = nn.Linear(output_dim, 1)


    def forward_internal(self, x):
        for cell in self.cells:
            x = cell(x)
        return x

    def forward(self, x):
        x = self.forward_internal(x)
        return self.linear(x)

    def train_on_batch(self, x, y_true, optimizer, loss_fn):
        self.train()
        optimizer.zero_grad()
        y_pred = torch.sigmoid(self.forward(x))
        loss = loss_fn(y_pred, y_true)
        loss.backward()
        optimizer.step()
        return loss.item()

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            preds = torch.sigmoid(self.forward(x))
        return (preds > 0.5).int().cpu().numpy().flatten()
