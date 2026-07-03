"""
conftest.py
-----------
Shared pytest fixtures for the spinq_vqe test suite.

All fixtures use n_cells=1 (N=3 sites) to keep circuit and VQE tests fast.
N=3 is the minimal Kagome unit cell — 1 triangle, 3 bonds.
"""

import numpy as np
import pytest

from spinq_vqe.kagome import heisenberg_kagome_hamiltonian, kagome_graph


# ---------------------------------------------------------------------------
# Graph fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def G1():
    """Kagome graph with 1 unit cell: 3 nodes, 3 edges."""
    return kagome_graph(n_cells=1)


@pytest.fixture(scope="session")
def G3():
    """Kagome graph with 3 unit cells: 9 nodes, 11 edges."""
    return kagome_graph(n_cells=3)


# ---------------------------------------------------------------------------
# Hamiltonian fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def H1(G1):
    """Heisenberg Hamiltonian on the 3-site graph (default params, normalize=True)."""
    return heisenberg_kagome_hamiltonian(G1)


@pytest.fixture(scope="session")
def H1_bare(G1):
    """Heisenberg Hamiltonian: J only (D=0, B=0), normalize=False."""
    return heisenberg_kagome_hamiltonian(G1, D=0.0, B=0.0, normalize=False)


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def edges1(G1):
    """Edge list for the 3-site graph."""
    return list(G1.edges())


@pytest.fixture(scope="session")
def edges3(G3):
    """Edge list for the 9-site graph."""
    return list(G3.edges())
