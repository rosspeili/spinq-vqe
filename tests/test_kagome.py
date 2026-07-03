"""
test_kagome.py
--------------
Unit tests for spinq_vqe.kagome — lattice graph construction and Hamiltonian builder.
"""

import numpy as np
import pytest

from spinq_vqe.kagome import (
    J_MN3SN_MEV,
    heisenberg_kagome_hamiltonian,
    kagome_graph,
    n_bonds,
    n_sites,
    sublattice_partition,
)


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


class TestKagomeGraph:
    def test_n_cells_1_nodes(self, G1):
        assert G1.number_of_nodes() == 3

    def test_n_cells_1_edges(self, G1):
        # One triangle: 3 intra-cell bonds
        assert G1.number_of_edges() == 3

    def test_n_cells_3_nodes(self, G3):
        assert G3.number_of_nodes() == 9

    def test_n_cells_3_edges(self, G3):
        # 3 triangles × 3 intra-bonds + 2 inter-cell bonds = 11
        assert G3.number_of_edges() == 11

    def test_sublattice_labels_present(self, G1):
        for _, data in G1.nodes(data=True):
            assert "sublattice" in data
            assert data["sublattice"] in {0, 1, 2}

    def test_sublattice_cycle(self, G3):
        # Site i should have sublattice = i % 3
        for node, data in G3.nodes(data=True):
            assert data["sublattice"] == node % 3

    def test_periodic_adds_one_bond(self):
        G_open = kagome_graph(n_cells=3, boundary="open")
        G_pbc = kagome_graph(n_cells=3, boundary="periodic")
        assert G_pbc.number_of_edges() == G_open.number_of_edges() + 1

    def test_single_cell_periodic_same_as_open(self):
        # boundary=="periodic" with n_cells=1 has no effect (guard: n_cells > 1)
        G_open = kagome_graph(n_cells=1, boundary="open")
        G_pbc = kagome_graph(n_cells=1, boundary="periodic")
        assert G_open.number_of_edges() == G_pbc.number_of_edges()

    def test_utility_n_sites(self, G1, G3):
        assert n_sites(G1) == 3
        assert n_sites(G3) == 9

    def test_utility_n_bonds(self, G1, G3):
        assert n_bonds(G1) == 3
        assert n_bonds(G3) == 11


# ---------------------------------------------------------------------------
# Hamiltonian construction
# ---------------------------------------------------------------------------


class TestHeisenbergHamiltonian:
    def test_returns_hamiltonian(self, H1):
        import pennylane as qp
        assert isinstance(H1, qp.Hamiltonian)

    def test_term_count_with_D(self, G1):
        # D=0.3 (default, non-zero): 3 bonds × 3 + 3 anisotropy = 12 terms
        H = heisenberg_kagome_hamiltonian(G1)
        assert len(H.coeffs) == 12

    def test_term_count_J_only(self, G1):
        # D=0, B=0: only exchange — 3 bonds × 3 terms = 9
        H = heisenberg_kagome_hamiltonian(G1, D=0.0, B=0.0)
        assert len(H.coeffs) == 9

    def test_term_count_with_field(self, G1):
        # D=0, B=1: exchange + Zeeman = 9 + 3 = 12
        H = heisenberg_kagome_hamiltonian(G1, D=0.0, B=1.0)
        assert len(H.coeffs) == 12

    def test_normalize_scales_coeffs(self, G1):
        H_norm = heisenberg_kagome_hamiltonian(G1, D=0.0, B=0.0, normalize=True)
        H_raw = heisenberg_kagome_hamiltonian(G1, D=0.0, B=0.0, normalize=False)
        n = G1.number_of_nodes()  # 3
        # Normalized exchange coeff = J_eff / n; raw = J_eff
        ratio = H_raw.coeffs[0] / H_norm.coeffs[0]
        assert abs(ratio - n) < 1e-10

    def test_zero_coupling_no_exchange(self, G1):
        H = heisenberg_kagome_hamiltonian(G1, J=0.0, D=0.0, B=0.0)
        # All coefficients should be zero (empty or zero coeffs)
        assert all(abs(c) < 1e-12 for c in H.coeffs)

    def test_coeffs_are_finite(self, H1):
        assert all(np.isfinite(c) for c in H1.coeffs)


# ---------------------------------------------------------------------------
# Sublattice partition
# ---------------------------------------------------------------------------


class TestSublatticePartition:
    def test_partition_keys(self, G1):
        p = sublattice_partition(G1)
        assert set(p.keys()) == {0, 1, 2}

    def test_partition_covers_all_sites(self, G1):
        p = sublattice_partition(G1)
        all_sites = sorted(p[0] + p[1] + p[2])
        assert all_sites == list(range(G1.number_of_nodes()))

    def test_partition_sizes_equal_for_n_cells_3(self, G3):
        # 9 sites / 3 sublattices = 3 each
        p = sublattice_partition(G3)
        assert len(p[0]) == len(p[1]) == len(p[2]) == 3
