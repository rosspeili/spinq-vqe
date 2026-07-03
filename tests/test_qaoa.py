"""
test_qaoa.py
------------
Unit tests for spinq_vqe.qaoa — Hamiltonian construction, circuit, and optimizer.

Uses N=4 materials (4 qubits) with very few optimizer steps to stay fast.
We verify structure, types, and constraint satisfaction — not convergence quality.
"""

import numpy as np
import pytest

from spinq_vqe.qaoa import (
    QAOAResult,
    build_cost_hamiltonian,
    build_mixer_hamiltonian,
    classical_greedy,
    run_qaoa,
)

# Small synthetic θ_SH dataset: 4 materials
THETA_SH_4 = np.array([0.35, -0.33, 0.40, 0.08])


# ---------------------------------------------------------------------------
# Hamiltonian builders
# ---------------------------------------------------------------------------


class TestBuildCostHamiltonian:
    def test_returns_hamiltonian(self):
        import pennylane as qp
        H = build_cost_hamiltonian(THETA_SH_4, k=2)
        assert isinstance(H, qp.Hamiltonian)

    def test_coeffs_finite(self):
        H = build_cost_hamiltonian(THETA_SH_4, k=2)
        assert all(np.isfinite(c) for c in H.coeffs)

    def test_larger_lam_larger_constraint_coeffs(self):
        H_low = build_cost_hamiltonian(THETA_SH_4, k=2, lam=1.0)
        H_high = build_cost_hamiltonian(THETA_SH_4, k=2, lam=10.0)
        # ZZ cross terms scale with lam — the maximum abs coeff should be larger
        assert max(abs(c) for c in H_high.coeffs) > max(abs(c) for c in H_low.coeffs)


class TestBuildMixerHamiltonian:
    def test_returns_hamiltonian(self):
        import pennylane as qp
        H = build_mixer_hamiltonian(4)
        assert isinstance(H, qp.Hamiltonian)

    def test_n_terms_equals_n_materials(self):
        H = build_mixer_hamiltonian(4)
        assert len(H.coeffs) == 4

    def test_all_unit_coeffs(self):
        H = build_mixer_hamiltonian(4)
        assert all(abs(c - 1.0) < 1e-10 for c in H.coeffs)


# ---------------------------------------------------------------------------
# QAOA run
# ---------------------------------------------------------------------------


class TestRunQAOA:
    @pytest.fixture(scope="class")
    @classmethod
    def result(cls):
        return run_qaoa(
            THETA_SH_4, k=2, p=1,
            n_optimizer_steps=15,
            n_seeds=1,
            verbose=False,
        )

    def test_returns_qaoa_result(self, result):
        assert isinstance(result, QAOAResult)

    def test_energy_finite(self, result):
        assert np.isfinite(result.energy)

    def test_gamma_shape(self, result):
        assert result.gamma.shape == (1,)  # p=1

    def test_beta_shape(self, result):
        assert result.beta.shape == (1,)   # p=1

    def test_selected_indices_count(self, result):
        # Should select exactly k=2 materials
        assert len(result.selected_indices) == 2

    def test_selected_indices_valid(self, result):
        N = len(THETA_SH_4)
        for idx in result.selected_indices:
            assert 0 <= idx < N

    def test_selected_theta_sh_finite(self, result):
        assert np.isfinite(result.selected_theta_sh)

    def test_metadata_fields(self, result):
        assert result.p == 1
        assert result.k == 2
        assert result.n_materials == 4

    def test_k_ge_n_raises(self):
        with pytest.raises(ValueError, match="k="):
            run_qaoa(THETA_SH_4, k=4, p=1, n_optimizer_steps=5, n_seeds=1, verbose=False)


# ---------------------------------------------------------------------------
# Classical baseline
# ---------------------------------------------------------------------------


class TestClassicalGreedy:
    def test_returns_indices(self):
        indices = classical_greedy(THETA_SH_4, k=2)
        assert len(indices) == 2

    def test_selects_top_k(self):
        # Greedy should select the two largest θ_SH: indices 2 (0.40) and 0 (0.35)
        indices = set(classical_greedy(THETA_SH_4, k=2))
        assert indices == {0, 2}

    def test_k_1_selects_best(self):
        idx = classical_greedy(THETA_SH_4, k=1)
        assert idx == [2]  # CrTe2 has max θ_SH = 0.40
