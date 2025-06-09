# constants.py

# Default features used by all cells
DEFAULT_FEATURES = ["time_ms", "voltage_mV", "input_current_nA"]

# Folder where data CSVs live
DATASET_DIR = "dataset_cerebelo"

# Folder where trained models are saved
MODEL_DIR = "models_cerebelo"

# List of all known cell types (for loops, training, etc.)
CELL_NAMES = ["granule_lif", "purkinje_hh_dinamico", "deep_nuclei_lif", "stellate_lif", "basket_lif", "golgi_lif", "climbing_fiber", "mossy_fiber"]