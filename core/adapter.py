import torch
import torch.nn as nn

class InputToCerebellarAdapter(nn.Module):
    def __init__(self, input_dim, adapted_dim=None):
        super().__init__()
        if adapted_dim is None:
            raise ValueError("adapted_dim must be specified or inferred from the cerebellar cell")
        self.fc = nn.Linear(input_dim, adapted_dim)

    def forward(self, x):
        return self.fc(x)
