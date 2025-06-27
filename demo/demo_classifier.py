import torch
import pandas as pd
import numpy as np
import os
import glob
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
from core.classifier import CerebellarANNClassifier
from cells.cell_types import Granule, Purkinje, DeepNuclei, Basket, MossyFiber, ClimbingFiber, Stellate

#conf
SEED = 42
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#data load
data_dir = "./demo/MovementAAL/dataset"
input_files = sorted(glob.glob(os.path.join(data_dir, "MovementAAL_RSS_*.csv")))
targets_df = pd.read_csv(os.path.join(data_dir, "MovementAAL_target.csv"))


X = []
y = []
for file in input_files:
    seq_id = int(file.split("_")[-1].split(".")[0])
    sequence = pd.read_csv(file).values
    try:
        label = targets_df[targets_df["#sequence_ID"] == seq_id][" class_label"].values[0]
    except IndexError:
        continue
    X.append(sequence.mean(axis=0))
    y.append(1 if label == 1 else 0)

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32).reshape(-1, 1)
print("Class count:", dict(zip(*np.unique(y, return_counts=True))))

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)
y_tensor = torch.tensor(y, dtype=torch.float32).to(DEVICE)

#k-fold
k = 10
fold_size = len(X_tensor) // k
indices = torch.randperm(len(X_tensor), generator=torch.Generator().manual_seed(SEED))
all_preds = []
all_targets = []

start_time = time.time()
for fold in range(k):
    print(f"\nFold {fold + 1}/{k}")

    val_idx = indices[fold * fold_size:(fold + 1) * fold_size]
    train_idx = torch.cat((indices[:fold * fold_size], indices[(fold + 1) * fold_size:]))

    X_train, y_train = X_tensor[train_idx], y_tensor[train_idx]
    X_val, y_val = X_tensor[val_idx], y_tensor[val_idx]

    model = CerebellarANNClassifier(optimizer_class=torch.optim.AdamW, loss_fn=torch.nn.BCELoss(), lr=0.001, epochs=50)
    model.add_cell(Granule(plasticity='STDP', inhibition=True))
    model.add_cell(Purkinje(plasticity='LTP', inhibition=True))
    model.add_cell(DeepNuclei(plasticity='LTP', inhibition=True))
    model.to(DEVICE)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    all_preds.extend(y_pred)
    all_targets.extend(y_val.cpu().numpy().flatten())

end_time = time.time()

# Evaluación
y_true = np.array(all_targets)
y_pred = np.array(all_preds)
print("\nClassification Report (10-fold CV):")
print(classification_report(y_true, y_pred, digits=4))
print("Total training + evaluation time: {:.2f} seconds".format(end_time - start_time))

# Matriz de Confusión
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix (10-fold CV)")
plt.grid(False)
plt.tight_layout()
plt.savefig("confusion_matrix_kfold.png")
plt.close()

# Curva ROC
fpr, tpr, _ = roc_curve(y_true, y_pred)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (10-fold CV)")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("roc_curve_kfold.png")
plt.close()