"""
test_entanglement.py
--------------------
Unit tests for spinq_vqe.entanglement — reduced density matrix, Von Neumann
entropy, and mutual information.

Uses synthetic statevectors with known entanglement properties as ground truth.
"""

import numpy as np
import pytest

from spinq_vqe.entanglement import (
    reduced_density_matrix,
    von_neumann_entropy,
)


# ---------------------------------------------------------------------------
# Helpers — known statevectors
# ---------------------------------------------------------------------------


def product_state(n: int) -> np.ndarray:
    """|0...0⟩ — fully separable, zero entanglement."""
    sv = np.zeros(2**n, dtype=complex)
    sv[0] = 1.0
    return sv


def bell_pair_extended(n: int) -> np.ndarray:
    """(|00...0⟩ + |11...1⟩) / √2 — maximally entangled across any bipartition."""
    sv = np.zeros(2**n, dtype=complex)
    sv[0] = 1 / np.sqrt(2)
    sv[-1] = 1 / np.sqrt(2)
    return sv


# ---------------------------------------------------------------------------
# Reduced density matrix
# ---------------------------------------------------------------------------


class TestReducedDensityMatrix:
    def test_shape_single_qubit(self):
        sv = product_state(3)
        rho = reduced_density_matrix(sv, subsystem=[0], n_sites=3)
        assert rho.shape == (2, 2)

    def test_shape_two_qubits(self):
        sv = product_state(4)
        rho = reduced_density_matrix(sv, subsystem=[0, 1], n_sites=4)
        assert rho.shape == (4, 4)

    def test_trace_equals_one(self):
        sv = bell_pair_extended(3)
        rho = reduced_density_matrix(sv, subsystem=[0], n_sites=3)
        assert abs(np.trace(rho) - 1.0) < 1e-10

    def test_hermitian(self):
        sv = bell_pair_extended(3)
        rho = reduced_density_matrix(sv, subsystem=[0, 1], n_sites=3)
        np.testing.assert_allclose(rho, rho.conj().T, atol=1e-12)

    def test_positive_semidefinite(self):
        sv = bell_pair_extended(4)
        rho = reduced_density_matrix(sv, subsystem=[0, 1], n_sites=4)
        eigvals = np.linalg.eigvalsh(rho)
        assert np.all(eigvals >= -1e-12)

    def test_pure_state_rho_squared_equals_rho(self):
        # For a pure global state, ρ_A² = ρ_A iff subsystem A is not entangled
        sv = product_state(3)
        rho = reduced_density_matrix(sv, subsystem=[0], n_sites=3)
        np.testing.assert_allclose(rho @ rho, rho, atol=1e-12)


# ---------------------------------------------------------------------------
# Von Neumann entropy
# ---------------------------------------------------------------------------


class TestVonNeumannEntropy:
    def test_zero_for_pure_state(self):
        sv = product_state(3)
        rho = reduced_density_matrix(sv, subsystem=[0], n_sites=3)
        S = von_neumann_entropy(rho)
        assert abs(S) < 1e-10

    def test_maximal_for_bell_pair(self):
        # (|00⟩ + |11⟩)/√2 — single-qubit subsystem: S = 1 bit
        sv = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        rho = reduced_density_matrix(sv, subsystem=[0], n_sites=2)
        S = von_neumann_entropy(rho, base=2.0)
        assert abs(S - 1.0) < 1e-8

    def test_non_negative(self):
        sv = bell_pair_extended(4)
        rho = reduced_density_matrix(sv, subsystem=[0, 1], n_sites=4)
        assert von_neumann_entropy(rho) >= -1e-12

    def test_bounded_by_log_dim(self):
        # S ≤ log2(dim_A) for base=2
        sv = bell_pair_extended(4)
        n_A = 2
        rho = reduced_density_matrix(sv, subsystem=[0, 1], n_sites=4)
        S = von_neumann_entropy(rho, base=2.0)
        assert S <= np.log2(2**n_A) + 1e-8

    def test_base_e_vs_base_2(self):
        sv = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        rho = reduced_density_matrix(sv, subsystem=[0], n_sites=2)
        S2 = von_neumann_entropy(rho, base=2.0)
        Se = von_neumann_entropy(rho, base=np.e)
        assert abs(S2 * np.log(2) - Se) < 1e-8
