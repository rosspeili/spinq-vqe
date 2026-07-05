"""
test_vqe.py
-----------
Unit tests for spinq_vqe.vqe — COBYLA and Adam VQE runners.

All tests use N=3 (1 Kagome unit cell) with very few evaluations to stay fast.
We only verify that the optimizer runs, returns the correct result structure,
and moves energy in the right direction — not that it converges to the exact
ground state (N=3 is too small for a meaningful convergence test).
"""

import numpy as np
import pytest

from spinq_vqe.ansatz import hea_ansatz, hva_ansatz, init_params
from spinq_vqe.vqe import (
    VQEMultiSeedResult,
    VQEResult,
    run_vqe,
    run_vqe_cobyla,
    run_vqe_cobyla_multi_seed,
    seed_statistics,
)


# ---------------------------------------------------------------------------
# VQEResult dataclass
# ---------------------------------------------------------------------------


class TestVQEResult:
    def test_fields_exist(self):
        r = VQEResult(
            energy=-0.5,
            params=np.zeros(3),
            energy_history=[-0.1, -0.5],
            gradient_variance_history=[0.01],
            n_steps=2,
            converged=False,
        )
        assert r.energy == -0.5
        assert r.n_steps == 2
        assert r.statevector is None   # default
        assert r.ansatz == ""          # default


# ---------------------------------------------------------------------------
# COBYLA runner
# ---------------------------------------------------------------------------


class TestRunVQECobyla:
    def test_returns_vqe_result(self, H1_bare, edges1):
        params = init_params("hea", n_sites=3, depth=1, seed=0)
        result = run_vqe_cobyla(
            hamiltonian=H1_bare,
            ansatz_fn=hea_ansatz,
            init_params=params,
            n_sites=3,
            ansatz_name="hea",
            n_evals=30,
            verbose=False,
            depth=1,
            edges=edges1,
        )
        assert isinstance(result, VQEResult)

    def test_energy_is_finite(self, H1_bare, edges1):
        params = init_params("hea", n_sites=3, depth=1, seed=1)
        result = run_vqe_cobyla(
            H1_bare, hea_ansatz, params, 3,
            n_evals=30, verbose=False,
            depth=1, edges=edges1,
        )
        assert np.isfinite(result.energy)

    def test_energy_history_nonempty(self, H1_bare, edges1):
        params = init_params("hea", n_sites=3, depth=1, seed=2)
        result = run_vqe_cobyla(
            H1_bare, hea_ansatz, params, 3,
            n_evals=30, verbose=False,
            depth=1, edges=edges1,
        )
        assert len(result.energy_history) > 0

    def test_best_energy_le_first_eval(self, H1_bare, edges1):
        params = init_params("hea", n_sites=3, depth=1, seed=3)
        result = run_vqe_cobyla(
            H1_bare, hea_ansatz, params, 3,
            n_evals=50, verbose=False,
            depth=1, edges=edges1,
        )
        # best energy should be ≤ the first evaluation
        assert result.energy <= result.energy_history[0] + 1e-8

    def test_gradient_variance_history_empty(self, H1_bare, edges1):
        # COBYLA is gradient-free — this list should always be empty
        params = init_params("hea", n_sites=3, depth=1, seed=4)
        result = run_vqe_cobyla(
            H1_bare, hea_ansatz, params, 3,
            n_evals=20, verbose=False,
            depth=1, edges=edges1,
        )
        assert result.gradient_variance_history == []

    def test_n_steps_matches_history(self, H1_bare, edges1):
        params = init_params("hea", n_sites=3, depth=1, seed=5)
        result = run_vqe_cobyla(
            H1_bare, hea_ansatz, params, 3,
            n_evals=25, verbose=False,
            depth=1, edges=edges1,
        )
        assert result.n_steps == len(result.energy_history)

    def test_statevector_shape(self, H1_bare, edges1):
        params = init_params("hea", n_sites=3, depth=1, seed=6)
        result = run_vqe_cobyla(
            H1_bare, hea_ansatz, params, 3,
            n_evals=20, verbose=False,
            return_statevector=True,
            depth=1, edges=edges1,
        )
        assert result.statevector is not None
        assert result.statevector.shape == (8,)  # 2^3

    def test_statevector_normalized(self, H1_bare, edges1):
        params = init_params("hea", n_sites=3, depth=1, seed=7)
        result = run_vqe_cobyla(
            H1_bare, hea_ansatz, params, 3,
            n_evals=20, verbose=False,
            return_statevector=True,
            depth=1, edges=edges1,
        )
        norm = np.sum(np.abs(result.statevector) ** 2)
        assert abs(norm - 1.0) < 1e-6

    def test_metadata_fields(self, H1_bare, edges1):
        params = init_params("hea", n_sites=3, depth=1, seed=8)
        result = run_vqe_cobyla(
            H1_bare, hea_ansatz, params, 3,
            ansatz_name="hea",
            n_evals=20, verbose=False,
            depth=1, edges=edges1,
        )
        assert result.optimizer == "cobyla"
        assert result.ansatz == "hea"
        assert result.n_sites == 3
        assert result.n_params == 3


# ---------------------------------------------------------------------------
# Adam runner (gradient-based — diagnostic use)
# ---------------------------------------------------------------------------


class TestRunVQEAdam:
    def test_returns_vqe_result(self, H1_bare, edges1):
        params = init_params("hea", n_sites=3, depth=1, seed=10)
        result = run_vqe(
            H1_bare, hea_ansatz, params, 3,
            n_steps=5, verbose=False,
            depth=1, edges=edges1,
        )
        assert isinstance(result, VQEResult)

    def test_grad_var_history_nonempty(self, H1_bare, edges1):
        params = init_params("hea", n_sites=3, depth=1, seed=11)
        result = run_vqe(
            H1_bare, hea_ansatz, params, 3,
            n_steps=5, verbose=False,
            depth=1, edges=edges1,
        )
        assert len(result.gradient_variance_history) == 5

    def test_optimizer_field(self, H1_bare, edges1):
        params = init_params("hea", n_sites=3, depth=1, seed=12)
        result = run_vqe(
            H1_bare, hea_ansatz, params, 3,
            n_steps=3, verbose=False,
            depth=1, edges=edges1,
        )
        assert result.optimizer == "adam"


# ---------------------------------------------------------------------------
# Seed statistics and multi-seed COBYLA
# ---------------------------------------------------------------------------


class TestSeedStatistics:
    def test_single_seed(self):
        stats = seed_statistics([-1.0])
        assert stats.n_seeds == 1
        assert stats.mean_energy == -1.0
        assert stats.std_energy == 0.0
        assert stats.min_energy == -1.0
        assert stats.max_energy == -1.0

    def test_multiple_seeds(self):
        stats = seed_statistics([-1.2, -1.0, -1.4])
        assert stats.n_seeds == 3
        assert abs(stats.mean_energy - (-1.2)) < 1e-12
        assert stats.min_energy == -1.4
        assert stats.max_energy == -1.0
        assert stats.std_energy > 0.0

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            seed_statistics([])


class TestRunVQECobylaMultiSeed:
    def test_returns_multi_seed_result(self, H1_bare, edges1):
        seeds = [0, 1, 2]

        def init_fn(seed: int) -> np.ndarray:
            return init_params("hea", n_sites=3, depth=1, seed=seed)

        result = run_vqe_cobyla_multi_seed(
            hamiltonian=H1_bare,
            ansatz_fn=hea_ansatz,
            init_params_fn=init_fn,
            n_sites=3,
            seeds=seeds,
            n_evals=25,
            verbose=False,
            depth=1,
            edges=edges1,
        )
        assert isinstance(result, VQEMultiSeedResult)
        assert len(result.runs) == 3
        assert result.seeds == seeds
        assert result.statistics.n_seeds == 3
        assert result.best.energy == min(r.energy for r in result.runs)

    def test_best_has_statevector_when_requested(self, H1_bare, edges1):
        result = run_vqe_cobyla_multi_seed(
            hamiltonian=H1_bare,
            ansatz_fn=hea_ansatz,
            init_params_fn=lambda s: init_params("hea", 3, depth=1, seed=s),
            n_sites=3,
            seeds=[4, 5],
            n_evals=20,
            return_statevector=True,
            verbose=False,
            depth=1,
            edges=edges1,
        )
        assert result.best.statevector is not None
        assert result.best.statevector.shape == (8,)
        assert all(r.statevector is None for r in result.runs)

    def test_empty_seeds_raises(self, H1_bare, edges1):
        with pytest.raises(ValueError, match="at least one"):
            run_vqe_cobyla_multi_seed(
                H1_bare,
                hea_ansatz,
                lambda s: init_params("hea", 3, depth=1, seed=s),
                3,
                seeds=[],
                verbose=False,
                depth=1,
                edges=edges1,
            )
