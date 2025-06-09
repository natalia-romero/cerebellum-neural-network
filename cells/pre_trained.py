from Cerebellar_Class import NeuronaCerebelarKAN
from core.constants import CELL_NAMES, FEATURES, DATA_DIR
import os

for name in CELL_NAMES:
    path_csv = os.path.join(DATA_DIR, f"{name}_light.csv")
    if os.path.exists(path_csv):
        print(f"\n🚀 Pretraining cell: {name}")
        cell = NeuronaCerebelarKAN(nombre_celula=name, columnas_features=FEATURES)
        cell.ruta_datos_csv = path_csv
        cell.configurar_entrenamiento_personalizado(epochs_per_phase=10)
        cell.entrenar_modelo(forzar_reentrenamiento=True)
    else:
        print(f"⚠️  Missing data file for {name}: {path_csv}")
