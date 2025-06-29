import torch
import pandas as pd
import numpy as np
import os
import glob
import time
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, auc

from core.classifier import CINNClassifier
from cells.cell_types import Granule, Purkinje, DeepNuclei

SEED = 42
FOLDS = 10
EPOCHS = 50
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#data
data_dir = "./demo/MovementAAL/dataset"
input_files = sorted(glob.glob(os.path.join(data_dir, "MovementAAL_RSS_*.csv")))
targets_df = pd.read_csv(os.path.join(data_dir, "MovementAAL_target.csv"))

X, y = [], []
for file in input_files:
    seq_id = int(file.split("_")[-1].split(".")[0])
    sequence = pd.read_csv(file).values
    try:
        label = targets_df[targets_df["#sequence_ID"] == seq_id][" class_label"].values[0]
        X.append(sequence.mean(axis=0))
        y.append(1 if label == 1 else 0)
    except IndexError:
        continue

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32).reshape(-1, 1)

scaler = StandardScaler()
X = scaler.fit_transform(X)

#KFold
output_dir = "results_kfold"
os.makedirs(output_dir, exist_ok=True)

fold_size = len(X) // FOLDS
metrics = {"acc": [], "prec": [], "rec": [], "f1": []}
y_true_all, y_pred_all = [], []
fold_losses = []

print("Cross Validation 10-fold...")
start_time = time.time()

for fold in range(FOLDS):
    print(f"Fold {fold + 1}/{FOLDS}...")
    start, end = fold * fold_size, (fold + 1) * fold_size
    X_val, y_val = X[start:end], y[start:end]
    X_train = np.concatenate([X[:start], X[end:]])
    y_train = np.concatenate([y[:start], y[end:]])

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(DEVICE)

    model = CINNClassifier()
    model.add_cell(Granule(plasticity='STDP', inhibition=True))
    model.add_cell(Purkinje(plasticity='LTP', inhibition=False))
    model.add_cell(DeepNuclei(plasticity='LTD', inhibition=True))
    model.finalize()
    model.to(DEVICE)

    optimizer = torch.optim.AdamW(model.get_trainable_parameters(), lr=0.01)
    loss_fn = torch.nn.BCELoss()

    losses = model.fit(X_train_tensor, y_train_tensor, optimizer, loss_fn, epochs=EPOCHS)
    fold_losses.append(losses)

    y_pred = model.predict(X_val_tensor)
    y_pred_bin = (y_pred > 0.5).astype(int)
    y_true = y_val_tensor.cpu().numpy().astype(int).flatten()

    y_true_all.extend(y_true)
    y_pred_all.extend(y_pred_bin)

    metrics["acc"].append(accuracy_score(y_true, y_pred_bin))
    metrics["prec"].append(precision_score(y_true, y_pred_bin))
    metrics["rec"].append(recall_score(y_true, y_pred_bin))
    metrics["f1"].append(f1_score(y_true, y_pred_bin))

training_time = time.time() - start_time

#results
for i, losses in enumerate(fold_losses):
    plt.figure()
    plt.plot(range(1, len(losses)+1), losses, marker='o')
    plt.title(f"Fold {i+1} Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"fold_{i+1}_loss_curve.png"))
    plt.close()

print("\nAvg metrics per 10-Fold CV:")
summary_lines = []
for k, v in metrics.items():
    line = f"{k.upper()}: {np.mean(v):.4f} - {np.std(v):.4f}"
    print(line)
    summary_lines.append(line)

report_str = classification_report(y_true_all, y_pred_all, digits=4)
print("\nClassification report:")
print(report_str)

with open(os.path.join(output_dir, "evaluation_summary.txt"), "w") as f:
    f.write("Avg metrics per 10-Fold CV:\n")
    for line in summary_lines:
        f.write(line + "\n")
    f.write("\nClassification Report:\n")
    f.write(report_str)
    f.write(f"\nTotal Training Time: {training_time:.2f} seconds\n")

cm = confusion_matrix(y_true_all, y_pred_all)
plt.figure()
ConfusionMatrixDisplay(confusion_matrix=cm).plot()
plt.title("CINN Confusion Matrix - Indoor User Movement Dataset")
plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
plt.close()

fpr, tpr, _ = roc_curve(y_true_all, y_pred_all)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "roc_curve.png"))
plt.close()
