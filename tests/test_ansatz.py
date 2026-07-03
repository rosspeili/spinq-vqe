"""
test_ansatz.py
--------------
Unit tests for spinq_vqe.ansatz — HEA, HVA, MERA parameter counts and circuits.
"""

import numpy as np
import pennylane as qp
import pytest

from spinq_vqe.ansatz import (
    hea_ansatz,
    hea_n_params,
    hva_ansatz,
    hva_n_params,
    init_params,
    mera_ansatz,
    mera_n_params,
)


# ---------------------------------------------------------------------------
# Parameter count
# ---------------------------------------------------------------------------


class TestParamCounts:
    def test_hea_n_params_basic(self):
        assert hea_n_params(n_sites=9, depth=3) == 27

    def test_hea_n_params_small(self):
        assert hea_n_params(n_sites=3, depth=1) == 3

    def test_hea_n_params_linear(self):
        # depth * n_sites
        for d in [1, 2, 5]:
            assert hea_n_params(n_sites=6, depth=d) == 6 * d

    def test_hva_n_params_basic(self):
        assert hva_n_params(depth=6) == 18

    def test_hva_n_params_linear(self):
        # 3 per layer (shared θ_XX, θ_YY, θ_ZZ)
        for d in [1, 3, 10]:
            assert hva_n_params(depth=d) == 3 * d

    def test_mera_n_params_1cell(self, G1):
        # Scale1: 4×3=12, Scale2: 4×min(1,1)=4, Final: 3 → 19
        assert mera_n_params(G1) == 19

    def test_mera_n_params_3cell(self, G3):
        # Scale1: 4×11=44, Scale2: 4×min(3,3)=12, Final: 9 → 65
        assert mera_n_params(G3) == 65


# ---------------------------------------------------------------------------
# init_params
# ---------------------------------------------------------------------------


class TestInitParams:
    def test_hea_shape(self, G1):
        p = init_params("hea", n_sites=3, depth=1, seed=0)
        assert p.shape == (3,)

    def test_hva_shape(self):
        p = init_params("hva", n_sites=3, depth=2, seed=0)
        assert p.shape == (6,)

    def test_mera_shape(self, G1):
        p = init_params("mera", n_sites=3, G=G1, seed=0)
        assert p.shape == (19,)

    def test_seed_reproducible(self):
        p1 = init_params("hea", n_sites=6, depth=2, seed=7)
        p2 = init_params("hea", n_sites=6, depth=2, seed=7)
        np.testing.assert_array_equal(p1, p2)

    def test_different_seeds_different(self):
        p1 = init_params("hea", n_sites=6, depth=2, seed=1)
        p2 = init_params("hea", n_sites=6, depth=2, seed=2)
        assert not np.allclose(p1, p2)

    def test_scale_range(self):
        p = init_params("hea", n_sites=6, depth=3, seed=0, scale=0.1)
        assert np.all(np.abs(p) <= np.pi * 0.1 + 1e-10)

    def test_mera_requires_graph(self):
        with pytest.raises(ValueError, match="G"):
            init_params("mera", n_sites=3, G=None)

    def test_unknown_ansatz_raises(self):
        with pytest.raises(ValueError, match="Unknown ansatz"):
            init_params("unknown", n_sites=3)


# ---------------------------------------------------------------------------
# Circuit execution (N=3, trivial system)
# ---------------------------------------------------------------------------


class TestCircuitExecution:
    """Verify each ansatz circuit runs without error on a 3-qubit device."""

    def _run_circuit(self, ansatz_fn, params, n_sites, **kwargs):
        dev = qp.device("default.qubit", wires=n_sites)

        @qp.qnode(dev)
        def circuit(p):
            ansatz_fn(p, n_sites, **kwargs)
            return qp.state()

        return circuit(params)

    def test_hea_runs(self, edges1):
        params = init_params("hea", n_sites=3, depth=1, seed=42)
        sv = self._run_circuit(hea_ansatz, params, 3, depth=1, edges=edges1)
        assert sv.shape == (8,)
        assert abs(np.sum(np.abs(sv) ** 2) - 1.0) < 1e-6

    def test_hva_runs(self, edges1):
        params = init_params("hva", n_sites=3, depth=2, seed=42, scale=0.05)
        sv = self._run_circuit(hva_ansatz, params, 3, depth=2, edges=edges1)
        assert sv.shape == (8,)
        assert abs(np.sum(np.abs(sv) ** 2) - 1.0) < 1e-6

    def test_mera_runs(self, G1):
        params = init_params("mera", n_sites=3, G=G1, seed=42)
        sv = self._run_circuit(mera_ansatz, params, 3, G=G1)
        assert sv.shape == (8,)
        assert abs(np.sum(np.abs(sv) ** 2) - 1.0) < 1e-6
