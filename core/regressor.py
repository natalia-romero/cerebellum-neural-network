import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, r2_score

class CerebellarANNRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.cells = []
        self.linear = None  # Se define después con finalize()

    def add_cell(self, cell):
        self.cells.append(cell)

    def finalize(self):
        # Dummy input para inicializar la red
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
        return self.linear(x)

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


    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return self.forward(x).cpu().numpy().flatten()

    def get_trainable_parameters(self):
        params = []
        for cell in self.cells:
            params += [p for p in cell.model.parameters() if p.requires_grad]
        if self.linear is not None:
            params += [p for p in self.linear.parameters() if p.requires_grad]
        return params

    def evaluate(self, X_test, y_test, scaler_y=None, return_predictions=False, plot_path="regression_plot.png"):
        self.eval()
        with torch.no_grad():
            y_pred = self.predict(X_test)
            y_pred_np = y_pred.reshape(-1, 1)
            y_test_np = y_test.cpu().numpy().reshape(-1, 1)

            if scaler_y is not None:
                y_pred_np = scaler_y.inverse_transform(y_pred_np)
                y_test_np = scaler_y.inverse_transform(y_test_np)

            mse = mean_squared_error(y_test_np, y_pred_np)
            r2 = r2_score(y_test_np, y_pred_np)

            print("\nEvaluation Metrics:")
            print("MSE:", round(mse, 4))
            print("R2 :", round(r2, 4))
            print("RMSE:", round(mse ** 0.5, 4))

            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 6))
            plt.plot(y_test_np, label="Ground Truth", marker='o')
            plt.plot(y_pred_np, label="Prediction", marker='x')
            plt.title("Regression - Evaluation")
            plt.xlabel("Sample Index")
            plt.ylabel("Predicted Value")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()

            if return_predictions:
                return y_test_np, y_pred_np
