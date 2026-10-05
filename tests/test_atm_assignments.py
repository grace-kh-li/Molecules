from collections import Counter

import numpy as np
import pytest

from src.molecular_structure.CaNH2.CaNH2_package import CaNH2_molecule
from src.molecular_structure.RotationalStates import (
    diagonalize_ATM_Hamiltonian,
    rename_ATM_states,
)
from src.quantum_mechanics.Basis import QuantumState


@pytest.fixture(scope="module")
def molecule():
    return CaNH2_molecule(vibronic_states_to_include=("X", "A", "B"), N_range=(0, 3))


def assignment(state):
    q = state.quantum_numbers
    return q["elec"], q["N"], q["ka"], q["kc"], q["J"], q["m"]


def assert_complete_assignments(states):
    expected = []
    for elec in ("X", "A", "B"):
        for N in range(4):
            for ka in range(N + 1):
                for kc in ([N] if ka == 0 else [N - ka, N - ka + 1]):
                    for J in ([0.5] if N == 0 else [N - 0.5, N + 0.5]):
                        for m in np.arange(-J, J + 1):
                            expected.append((elec, N, ka, kc, J, m))
    assert Counter(map(assignment, states)) == Counter(expected)


def assert_eigenvectors(H, energies, states):
    vectors = np.column_stack([s.coeff for s in states])
    np.testing.assert_allclose(vectors.conj().T @ vectors, np.eye(len(states)), atol=2e-12)
    # Absolute residual in MHz, including the ~5e8 MHz electronic origins.
    np.testing.assert_allclose(H.matrix @ vectors, vectors * energies, rtol=0, atol=2e-6)


def assert_rotational_symmetry(states):
    basis = states[0].defining_basis
    lookup = {tuple(sorted(b.quantum_numbers.items())): i for i, b in enumerate(basis)}
    partners = [lookup[tuple(sorted(dict(b.quantum_numbers, k=-b.k).items()))] for b in basis]
    phases = np.array([(-1)**(b.N - b.k) for b in basis])
    for s in states:
        # Independent action of a pi rotation about c (a=z, c=y).
        np.testing.assert_allclose(phases * s.coeff[partners], (-1)**s.kc * s.coeff, atol=2e-12)
        assert sum(abs(c)**2 for c, b in zip(s.coeff, basis) if b.k % 2 != s.ka % 2) < 1e-20


def test_notebook_n3_ka1_selection_has_no_ka0_states(molecule):
    for kc in (2, 3):
        # Use the exact selection convention from CaNH2 simplified.ipynb.
        selected = [s for s in molecule.eigenstates
                    if s[0].elec == "X" and s[0].N == 3 and s.ka == 1 and s.kc == kc]
        assert len(selected) == 14
        for s in selected:
            populations = Counter()
            for b, c in zip(molecule.caseB_basis, s.coeff):
                populations[abs(b.k)] += abs(c)**2
            assert populations[1] > 0.999
            assert populations[0] < 1e-20


def test_all_zero_field_labels_and_doublet_symmetries(molecule):
    assert_complete_assignments(molecule.eigenstates)
    assert_rotational_symmetry(molecule.eigenstates)
    assert_eigenvectors(molecule.H, molecule.Es, molecule.eigenstates)
    for s in molecule.eigenstates:
        q = s.quantum_numbers
        assert s[0].N == s.N
        assert abs(s[0].k) == s.ka
        assert sum(abs(c)**2 for b, c in zip(molecule.caseB_basis, s.coeff)
                   if b.J != q["J"] or b.m != q["m"]) < 1e-20


def test_excited_doublet_labels_survive_reversed_energy_order(molecule):
    energies = {assignment(s): E for E, s in zip(molecule.Es, molecule.eigenstates)}
    # A-state spin-rotation inverts these doublets relative to the X state.
    for N, J in ((1, 1.5), (2, 2.5)):
        assert energies[("A", N, 1, N - 1, J, 0.5)] < energies[("A", N, 1, N, J, 0.5)]
        assert energies[("X", N, 1, N - 1, J, 0.5)] > energies[("X", N, 1, N, J, 0.5)]


def test_labels_do_not_depend_on_state_order_phase_or_display_sort(molecule):
    originals = molecule.eigenstates[::-1]
    copies = [QuantumState("unassigned", s.coeff * np.exp(0.37j * i), s.defining_basis)
              for i, s in enumerate(originals)]
    # Printing basis vectors reorders their quantum-number dictionaries.
    for b in molecule.caseB_basis:
        str(b)
    rename_ATM_states(copies)
    assert list(map(assignment, copies)) == list(map(assignment, originals))


@pytest.mark.parametrize("spin_rotation", [False, True])
def test_degenerate_m_levels_need_no_artificial_shift(molecule, spin_rotation):
    H = molecule.H_rot + molecule.H_offset
    if spin_rotation:
        H = H + molecule.H_SR
    Es, states = diagonalize_ATM_Hamiltonian(H)
    rename_ATM_states(states)
    assert_complete_assignments(states)
    assert_rotational_symmetry(states)
    assert_eigenvectors(H, Es, states)


@pytest.mark.parametrize("B", [(0, 0, 1), (1, 0, 0), (0, 1, 0)])
def test_zeeman_couplings_are_preserved(molecule, B):
    H = molecule.H
    for axis, field in zip(("x", "y", "z"), B):
        H = H + molecule.Zeeman_terms[axis] * field
    Es, states = diagonalize_ATM_Hamiltonian(H)
    rename_ATM_states(states)
    assert_rotational_symmetry(states)
    assert_eigenvectors(H, Es, states)
