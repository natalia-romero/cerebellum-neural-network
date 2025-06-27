import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin

class CerebellarANNClassifier(nn.Module, BaseEstimator, ClassifierMixin):
    def __init__(self, optimizer_class=torch.optim.AdamW, loss_fn=torch.nn.BCELoss(), lr=0.001, epochs=50):
        super().__init__()
        self.optimizer_class = optimizer_class
        self.loss_fn = loss_fn
        self.lr = lr
        self.epochs = epochs
        self.cells = []
        self.linear = None
        self.finalized = False
        self.optimizer = None

    def add_cell(self, cell):
        self.cells.append(cell)

    def finalize(self):
        if not self.finalized and self.cells:
            dummy_input = torch.zeros(1, self.cells[0].input_dim)
            with torch.no_grad():
                out = self.forward_internal(dummy_input)
            output_dim = out.shape[1]
            self.linear = nn.Linear(output_dim, 1)
            self.finalized = True

    def forward_internal(self, x):
        for cell in self.cells:
            x = cell(x)
        return x

    def forward(self, x):
        x = self.forward_internal(x)
        x = self.linear(x)
        return torch.sigmoid(x)

    def fit(self, X, y):
        self.finalize()
        params = [p for p in self.parameters() if p.requires_grad]
        if not params:
            raise ValueError("No trainable parameters found in model.")

        optimizer = torch.optim.AdamW(params, lr=self.lr)
        loss_fn = torch.nn.BCELoss()

        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

        for epoch in range(self.epochs):
            total_loss = 0.0
            for x_batch, y_batch in loader:
                optimizer.zero_grad()
                y_pred = self.forward(x_batch)
                loss = loss_fn(y_pred, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if epoch % 5 == 0 or epoch == self.epochs - 1:
                print(f"Epoch {epoch + 1}: Loss = {total_loss / len(loader):.4f}")

    def predict(self, X):
        self.eval()
        preds = []
        with torch.no_grad():
            for x in X:
                y_pred = self.forward(x.unsqueeze(0))
                preds.append(int(y_pred.item() > 0.5))
        return torch.tensor(preds).numpy()