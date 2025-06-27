import torch
import torch.nn as nn
from cells.Cerebellar_Class import NeuronaCerebelarKAN

class CerebellarCell(nn.Module):
    def __init__(self, cell_name, plasticity='STDP', inhibition=False, adaptive=False):
        super().__init__() 
        self.core = NeuronaCerebelarKAN(nombre_celula=cell_name)
        if not self.core.cargar_modelo():
            raise RuntimeError(f"Cell model '{cell_name}' could not be loaded.")
        self.model = self.core.modelo_kan_cargado

        # Inferir el input_dim desde cualquier parámetro de la red
        for param in self.model.parameters():
            if param.ndim >= 2:
                self.input_dim = param.shape[1]
                break
        else:
            raise RuntimeError(f"No se pudo inferir input_dim del modelo KAN '{cell_name}'.")


        self.plasticity = plasticity
        self.inhibition = inhibition
        self.adaptive = adaptive

        self.last_input = None
        self.last_output = None
        self.activity_trace = 0.0

    def __call__(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)  # for batch of size 1

        self.last_input = x.detach()
        # Ajuste de input dim automático
        expected_dim = self.model.width_in[0]
        current_dim = x.shape[1]
        if current_dim != expected_dim:
            if current_dim < expected_dim:
                padding = torch.zeros((x.shape[0], expected_dim - current_dim), device=x.device)
                x = torch.cat([x, padding], dim=1)
            else:
                x = x[:, :expected_dim]
        out = self.model(x)

        if self.adaptive:
            mean_act = out.mean().item()
            self.activity_trace = 0.9 * self.activity_trace + 0.1 * mean_act
            out = out - 0.1 * (self.activity_trace - 0.2)

        if self.inhibition:
            inhibition_level = torch.relu(self.last_input).mean().item()
            out = out - 0.1 * inhibition_level

        self.last_output = out.detach()
        return out.view(x.shape[0], -1)

    def apply_plasticity(self, error_signal=None):
        with torch.no_grad():
            if self.last_input is None:
                return

            batch_size = self.last_input.shape[0]
            num_basis = self.last_input.shape[1]
            grid = torch.linspace(0, 1, steps=num_basis).to(self.last_input.device)
            expanded_inputs = [1 - torch.abs(self.last_input - k) for k in grid]
            basis_stack = torch.stack(expanded_inputs, dim=-1).view(batch_size, -1)
            expanded_input = basis_stack[0]

            predicted = self.model(self.last_input).squeeze(0)
            delta = (
                error_signal + (predicted - predicted.detach())
                if error_signal is not None
                else predicted - predicted.detach()
            )

            for name, param in self.model.named_parameters():
                if name.endswith(".coef"):
                    out_dim, in_dim, num_grid = param.shape
                    expanded_input = self.last_input[0].repeat(out_dim, 1).unsqueeze(-1)
                    delta_term = delta.view(out_dim, 1, 1).expand_as(param)
                    update = 0.01 * (1 - torch.abs(torch.linspace(0, 1, num_grid, device=param.device) - 0.5))
                    update = update.view(1, 1, num_grid).expand_as(param)

                    if self.plasticity == 'LTD':
                        update = -update
                    elif self.plasticity == 'LTP':
                        update = +update
                    elif self.plasticity == 'STDP':
                        update = (param - param.detach())

                    param.data.copy_(
                        torch.clamp(param.data + update * delta_term, min=-1.0, max=1.0)
                    )