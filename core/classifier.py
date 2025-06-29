import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

class CINNClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.cells = []
        self.linear = None 

    def add_cell(self, cell):
        self.cells.append(cell)

    def finalize(self):
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.cells[0].input_dim)
            out = self.forward_internal(dummy_input)
        output_dim = out.shape[1]
        self.linear = nn.Linear(output_dim, 1)

    def forward_internal(self, x):
        for cell in self.cells:
            x = cell(x)
        return x

    def forward(self, x):
        x = self.forward_internal(x)
        return torch.sigmoid(self.linear(x))

    def get_trainable_parameters(self):
        params = []
        for cell in self.cells:
            params += [p for p in cell.model.parameters() if p.requires_grad]
        if self.linear is not None:
            params += [p for p in self.linear.parameters() if p.requires_grad]
        return params
    
    def fit(self, X, y, optimizer, loss_fn, epochs=50, batch_size=None):
        self.train()
        self.finalize()
        dataset = torch.utils.data.TensorDataset(X, y)
        if batch_size:
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        else:
            loader = [(X, y)]

        losses = []
        for epoch in range(epochs):
            total_loss = 0.0
            for x_batch, y_batch in loader:
                optimizer.zero_grad()
                y_pred = self.forward(x_batch)
                loss = loss_fn(y_pred, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(loader)
            losses.append(avg_loss)
            if epoch % 5 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch + 1}: Loss = {avg_loss:.4f}")
        return losses


    def predict(self, X):
        self.eval()
        with torch.no_grad():
            preds = self.forward(X)
            return (preds.cpu().numpy() > 0.5).astype(int).flatten()

    def evaluate(self, X, y_true, return_predictions=False):
        y_pred = self.predict(X)
        y_true_np = y_true.cpu().numpy().flatten()

        print("\nClassification Report:")
        print(classification_report(y_true_np, y_pred, digits=4))

        cm = confusion_matrix(y_true_np, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig("confusion_matrix.png")
        plt.close()

        fpr, tpr, _ = roc_curve(y_true_np, y_pred)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig("roc_curve.png")
        plt.close()

        if return_predictions:
            return y_true_np, y_pred