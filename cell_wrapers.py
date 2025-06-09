import torch
from cells.Cerebellar_Class import NeuronaCerebelarKAN

class CerebellarCell:
    def __init__(self, cell_name, plasticity='STDP', inhibition=False, adaptive=False):
        self.core = NeuronaCerebelarKAN(nombre_celula=cell_name)
        if not self.core.cargar_modelo():
            raise RuntimeError(f"Cell model '{cell_name}' could not be loaded.")
        self.model = self.core.modelo_kan_cargado
        self.plasticity = plasticity
        self.inhibition = inhibition
        self.adaptive = adaptive
        self.last_input = None
        self.last_output = None
        self.activity_trace = 0.0

    def __call__(self, x):
        self.last_input = x.detach()
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
            expanded_inputs = [1 - torch.abs(self.last_input - k) for k in self.model.grid_points]
            basis_stack = torch.stack(expanded_inputs, dim=-1).view(batch_size, -1)
            expanded_input = basis_stack[0]
            predicted = self.model(self.last_input.unsqueeze(0)).squeeze(0)
            delta = error_signal + (predicted - predicted.detach()) if error_signal is not None else predicted - predicted.detach()

            if self.plasticity == 'STDP':
                update = (predicted - predicted.detach()) * (expanded_input - expanded_input.mean())
                update = update.view(-1, 1) * 0.01
            else:
                update = 0.01 * (expanded_input.view(-1, 1) @ delta.view(1, -1))
                if self.plasticity == 'LTD':
                    update = -update
                elif self.plasticity == 'LTP':
                    update = +update

            self.model.coeffs.data.copy_(
                torch.clamp(self.model.coeffs.data + update, min=-1.0, max=1.0)
            )
