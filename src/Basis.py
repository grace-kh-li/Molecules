class QuantumState:
    def __init__(self):
        pass

class ElectronicState:
    def __init__(self, label, irrep_label=None, symm_group=None):
        self.label = label

        self.irrep_label = irrep_label
        self.symm_group = symm_group