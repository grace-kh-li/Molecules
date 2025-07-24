from src.quantum_mechanics.Basis import *
from ElectronicStates import *
from VibrationalStates import *
from RotationalStates import *
from AngularMomentum import *

class SeparableBasis(OrthogonalBasis):
    def __init__(self, e_basis, v_basis, r_basis, es_basis):
        self.e_basis = e_basis
        self.v_basis = v_basis
        self.r_basis = r_basis
        self.es_basis = es_basis
        product_basis = e_basis * v_basis * r_basis * es_basis
        super().__init__(product_basis.basis_vectors, "separable basis (exclude nuclear spin)")

class SeparableBasis_with_NS(OrthogonalBasis):
    def __init__(self, e_basis, v_basis, r_basis, es_basis, ns_basis):
        self.e_basis = e_basis
        self.v_basis = v_basis
        self.r_basis = r_basis
        self.es_basis = es_basis
        self.ei_basis = ns_basis
        product_basis = e_basis * v_basis * r_basis * es_basis * ns_basis
        super().__init__(product_basis.basis_vectors, "separable basis (with nuclear spin)")

