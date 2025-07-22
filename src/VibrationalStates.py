from Basis import *

class VibrationalState(BasisVector):
    def __init__(self, mode, n_excitation, group=None, irrep=None):
        self.mode = mode
        self.n_excitation = n_excitation
        self.irrep = irrep
        self.group = group
        super().__init__(f"{mode}_{n_excitation}")

class lStates(VibrationalState):
    def __init__(self, mode, n_excitation, l, group=None, irrep=None):
        super().__init__(mode, n_excitation, group, irrep)
        self.l = l
        assert n_excitation % 2 == l % 2