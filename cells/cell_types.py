from cell_wrapers import CerebellarCell

class Granule(CerebellarCell):
    def __init__(self, plasticity='STDP', inhibition=False, adaptive=False):
        super().__init__('granule_lif', plasticity, inhibition, adaptive)

class Purkinje(CerebellarCell):
    def __init__(self, plasticity='STDP', inhibition=False, adaptive=False):
        super().__init__('purkinje_hh_dinamico', plasticity, inhibition, adaptive)

class DeepNuclei(CerebellarCell):
    def __init__(self, plasticity='STDP', inhibition=False, adaptive=False):
        super().__init__('deep_nuclei_lif', plasticity, inhibition, adaptive)

class Stellate(CerebellarCell):
    def __init__(self, plasticity='STDP', inhibition=False, adaptive=False):
        super().__init__('stellate_lif', plasticity, inhibition, adaptive)

class Basket(CerebellarCell):
    def __init__(self, plasticity='STDP', inhibition=False, adaptive=False):
        super().__init__('basket_lif', plasticity, inhibition, adaptive)

class Golgi(CerebellarCell):
    def __init__(self, plasticity='STDP', inhibition=False, adaptive=False):
        super().__init__('golgi_lif', plasticity, inhibition, adaptive)

class ClimbingFiber(CerebellarCell):
    def __init__(self, plasticity='STDP', inhibition=False, adaptive=False):
        super().__init__('climbing_fiber', plasticity, inhibition, adaptive)
