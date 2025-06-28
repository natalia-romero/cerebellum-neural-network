import torch
import pandas as pd
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from core.classifier import CerebellarANNClassifier
from cells.cell_types import Granule, Purkinje, DeepNuclei, Basket, MossyFiber, ClimbingFiber, Stellate

SEED = 42
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=SEED)
X_train = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
y_train = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)
X_test = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
y_test = torch.tensor(y_test, dtype=torch.float32).to(DEVICE)

# Build model
model = CerebellarANNClassifier()
model.add_cell(Granule(plasticity='STDP', inhibition=True))
model.add_cell(Purkinje(plasticity='LTP', inhibition=False))
model.add_cell(DeepNuclei(plasticity='LTD', inhibition=True))
model.to(DEVICE)

optimizer = torch.optim.AdamW(model.get_trainable_parameters(), lr=0.01)
loss_fn = torch.nn.BCELoss()

# Train
print("\nTraining classifier...")
model.fit(X_train, y_train, optimizer, loss_fn, epochs=50)

# Evaluate
model.evaluate(X_test, y_test)