from src.quantum_mechanics.Basis import *
from src.quantum_mechanics.Operator import Operator


class VibronicState(BasisVector):
    def __init__(self, label, Lambda = 0, symmetry_group=None, irrep=None):
        self.Lambda = Lambda
        self.info = "VibronicState"
        super().__init__(label,symmetry_group=symmetry_group, irrep=irrep)
        self.quantum_numbers = {"elec": label}


class VibrationalState(BasisVector):
    def __init__(self, n_excitation_list, group=None, irrep=None):
        self.n_modes = len(n_excitation_list)
        self.excitations = n_excitation_list
        self.irrep = irrep
        self.group = group
        label = ""
        if self.n_modes == 1: # diatomic
            label = "v={}".format(self.excitations[0])
        elif self.n_modes == 3: # linear triatomic
            for n in self.excitations:
                label += str(n)
        else:
            for i, n in enumerate(self.excitations):
                if n > 0:
                    label += f"{i}_n "
            if label == "":
                label = "v=0"
        super().__init__(label)
        self.quantum_numbers = {"v": label}

class OffsetOperator(Operator):
    def __init__(self, basis, energy_dict):
        matrix = np.zeros((basis.dimension, basis.dimension), dtype=np.complex128)
        for i, b in enumerate(basis):
            matrix[i,i] = energy_dict[b.quantum_numbers["elec"]]
        super().__init__(basis, matrix)