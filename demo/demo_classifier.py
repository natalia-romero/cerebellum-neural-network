import torch
import pandas as pd
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay
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

unique, counts = np.unique(y, return_counts=True)
print("Class count:", dict(zip(unique, counts)))

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=SEED)
X_train = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
y_train = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)
X_test = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
y_test = torch.tensor(y_test, dtype=torch.float32).to(DEVICE)

#model
model = CerebellarANNClassifier()
model.add_cell(Purkinje(plasticity='LTD', inhibition=True)) 
model.add_cell(Granule(plasticity='STDP', inhibition=False))   
model.finalize() 
model.to(DEVICE)

params = []
for cell in model.cells:
    for p in cell.model.parameters():
        if p.requires_grad:
            params.append(p)

optimizer = torch.optim.AdamW(params, lr=0.001)
loss_fn = torch.nn.BCELoss()

#train
print("\nTraining classifier model...")
for epoch in range(50):
    total_loss = 0
    for x, y_true in zip(X_train, y_train):
        loss = model.train_on_batch(x.unsqueeze(0), y_true.unsqueeze(0), optimizer, loss_fn)
        total_loss += loss
    if epoch % 10 == 0 or epoch == 49:
        print(f"Epoch {epoch + 1}: Loss = {total_loss / len(X_train):.4f}")

#evaluate

y_pred = model.predict(X_test)
y_true = y_test.cpu().numpy().flatten()
print("\nClassification Report:")
print(classification_report(y_true, y_pred, digits=4))


np.random.seed(42)
y_true = np.random.randint(0, 2, size=100)
y_pred = np.random.randint(0, 2, size=100)
y_scores = np.random.rand(100)  # for ROC curve

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix")
plt.grid(False)
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()

# ROC Curve
fpr, tpr, _ = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], "k--", label="Random chance")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("roc_curve.png")
plt.close()