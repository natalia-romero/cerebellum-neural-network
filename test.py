import torch

model = torch.load("./models_cerebelo/kan_model_granule_lif.pt", map_location="cpu")

print("Atributos del modelo:", dir(model))
if hasattr(model, "input_id"):
    print("input_id:", model.input_id)
