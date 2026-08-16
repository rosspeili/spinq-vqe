"""
test_nqs.py
-----------
Tests for spinq_vqe.nqs (requires netket).
"""

from __future__ import annotations

import pytest

nk = pytest.importorskip("netket")

from spinq_vqe import nqs  # noqa: E402

ED_N9 = -1.4219039949999581


class TestNQSHamiltonian:
    def test_hamiltonian_matches_pennylane_n3(self):
        diff = nqs.validate_hamiltonian_against_pennylane(3)
        assert diff < 1e-10

    def test_hamiltonian_matches_pennylane_n4(self):
        diff = nqs.validate_hamiltonian_against_pennylane(4)
        assert diff < 1e-10


class TestNQSEnergy:
    def test_complex_rbm_n9_within_5_percent(self):
        """Acceptance: complex RBM recovers ED to <5% (typically ≪1%)."""
        result = nqs.run_nqs(
            3,
            model="rbm",
            n_iter=300,
            alpha=2.0,
            learning_rate=0.01,
            seed=42,
            backend="exact",
            complex_params=True,
            optimizer="adam",
            show_progress=False,
        )
        rel_err = abs(result.e0 - ED_N9) / abs(ED_N9) * 100
        assert result.n_sites == 9
        assert result.backend == "exact"
        assert rel_err < 5.0

    def test_real_rbm_plateaus_above_target(self):
        """Document why complex params are required on this strip."""
        result = nqs.run_nqs(
            3,
            model="rbm",
            n_iter=80,
            alpha=2.0,
            learning_rate=0.05,
            seed=42,
            backend="exact",
            complex_params=False,
            optimizer="sgd",
            show_progress=False,
        )
        rel_err = abs(result.e0 - ED_N9) / abs(ED_N9) * 100
        # Real RBM cannot encode Kagome signs — stays far above ED.
        assert rel_err > 20.0

    def test_save_and_load_csv(self, tmp_path):
        rows = [
            {
                "n_sites": 9,
                "n_cells": 3,
                "E_ED": f"{ED_N9:.10f}",
                "E_DMRG": f"{ED_N9:.10f}",
                "E_NQS_RBM": f"{ED_N9:.10f}",
                "E_NQS_RBM_err": "0.0",
                "E_NQS_MODPHASE": f"{ED_N9:.10f}",
                "E_NQS_MODPHASE_err": "0.0",
                "E_VQE": "-1.28456310",
                "NQS_RBM_error_vs_ED_pct": "0.0000",
                "NQS_MODPHASE_error_vs_ED_pct": "0.0000",
                "VQE_error_vs_ED_pct": "9.6589",
                "NQS_backend": "exact",
            }
        ]
        path = nqs.save_method_comparison_csv(rows, tmp_path / "method.csv")
        loaded = nqs.load_method_comparison_csv(path)
        assert len(loaded) == 1
        assert loaded[0]["n_sites"] == "9"
        assert float(loaded[0]["E_NQS_RBM"]) == pytest.approx(ED_N9, rel=1e-8)
