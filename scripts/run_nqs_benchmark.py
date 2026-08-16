"""
Generate NQS method-comparison CSV and figures for NB07.

Append-only: writes new artifacts under data/ and figures/. Does not modify
dmrg_*, scaling_energy.png, or other existing committed figures.

Run from repo root:
    pip install -e ".[nqs]"
    python scripts/run_nqs_benchmark.py
"""

from __future__ import annotations

import csv
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from spinq_vqe import nqs

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
FIG = REPO / "figures"

# N=9: exact ED available. N=12: FullSumState still practical (dim 4096).
N_CELLS = [3, 4]
N_ITER = 400
ALPHA = 2.0
LEARNING_RATE = 0.01
SEED = 42


def _load_vqe_energies() -> dict[int, float]:
    out: dict[int, float] = {}
    with (DATA / "vqe_scaling.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["E_VQE"] in ("N/A", ""):
                continue
            out[int(row["N"])] = float(row["E_VQE"])
    return out


def _load_ed_energies() -> dict[int, float]:
    out: dict[int, float] = {}
    with (DATA / "ed_reference_energies.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            out[int(row["n_sites"])] = float(row["E0_normalized"])
    with (DATA / "vqe_scaling.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["E_ED"] not in ("N/A", ""):
                out[int(row["N"])] = float(row["E_ED"])
    return out


def _load_dmrg_energies() -> dict[int, float]:
    out: dict[int, float] = {}
    path = DATA / "dmrg_reference_energies.csv"
    if not path.exists():
        return out
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            out[int(row["n_sites"])] = float(row["E0_normalized"])
    return out


def _pct_err(e: float, ref: float | None) -> str:
    if ref is None or ref == 0:
        return ""
    return f"{abs(e - ref) / abs(ref) * 100:.4f}"


def main() -> None:
    warnings.filterwarnings("ignore")
    FIG.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)

    print("Validating NetKet Hamiltonian against PennyLane at N=9...")
    diff = nqs.validate_hamiltonian_against_pennylane(3)
    print(f"  max |H_netket - H_pl| = {diff:.3e}")

    ed = _load_ed_energies()
    vqe = _load_vqe_energies()
    dmrg_e = _load_dmrg_energies()

    rbm_results: dict[int, nqs.NQSResult] = {}
    mod_results: dict[int, nqs.NQSResult] = {}
    rows: list[dict] = []

    for n_cells in N_CELLS:
        n_sites = 3 * n_cells
        print(f"\n=== N={n_sites} (n_cells={n_cells}) ===")

        print("  RBM (complex, alpha=2)...")
        rbm = nqs.run_nqs(
            n_cells,
            model="rbm",
            n_iter=N_ITER,
            alpha=ALPHA,
            learning_rate=LEARNING_RATE,
            seed=SEED,
            backend="exact",
            complex_params=True,
            optimizer="adam",
            show_progress=True,
        )
        rbm_results[n_sites] = rbm
        print(f"    E0={rbm.e0:.10f}  backend={rbm.backend}")

        print("  RBMModPhase (alpha=2)...")
        mod = nqs.run_nqs(
            n_cells,
            model="rbm_modphase",
            n_iter=N_ITER,
            alpha=ALPHA,
            learning_rate=LEARNING_RATE,
            seed=SEED,
            backend="exact",
            complex_params=True,
            optimizer="adam",
            show_progress=True,
        )
        mod_results[n_sites] = mod
        print(f"    E0={mod.e0:.10f}  backend={mod.backend}")

        e_ed = ed.get(n_sites)
        e_dmrg = dmrg_e.get(n_sites)
        e_vqe = vqe.get(n_sites)
        ref = e_ed if e_ed is not None else e_dmrg
        rows.append(
            {
                "n_sites": n_sites,
                "n_cells": n_cells,
                "E_ED": f"{e_ed:.10f}" if e_ed is not None else "",
                "E_DMRG": f"{e_dmrg:.10f}" if e_dmrg is not None else "",
                "E_NQS_RBM": f"{rbm.e0:.10f}",
                "E_NQS_RBM_err": f"{rbm.e0_err:.3e}",
                "E_NQS_MODPHASE": f"{mod.e0:.10f}",
                "E_NQS_MODPHASE_err": f"{mod.e0_err:.3e}",
                "E_VQE": f"{e_vqe:.10f}" if e_vqe is not None else "",
                "NQS_RBM_error_vs_ED_pct": _pct_err(rbm.e0, ref),
                "NQS_MODPHASE_error_vs_ED_pct": _pct_err(mod.e0, ref),
                "VQE_error_vs_ED_pct": _pct_err(e_vqe, ref) if e_vqe is not None else "",
                "NQS_backend": rbm.backend,
            }
        )

        # Persist convergence histories as CSV (npy is gitignored).
        for label, res in (("rbm", rbm), ("modphase", mod)):
            hist = np.asarray(res.energy_history, dtype=float)
            np.save(DATA / f"nqs_{label}_history_n{n_sites}.npy", hist)
            np.savetxt(
                DATA / f"nqs_{label}_history_n{n_sites}.csv",
                hist,
                delimiter=",",
                header="energy",
                comments="",
            )

    csv_path = nqs.save_method_comparison_csv(rows, DATA / "method_comparison.csv")
    print(f"\nSaved -> {csv_path.relative_to(REPO)}")

    # --- Figure 1: VMC / exact-NQS convergence at N=9 ---
    r9 = rbm_results[9]
    m9 = mod_results[9]
    e_ref = ed[9]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(r9.energy_history, color="#5B8FA8", lw=2, label="RBM (complex, α=2)")
    ax.plot(
        m9.energy_history,
        color="#C97B7B",
        lw=2,
        ls="--",
        label="RBMModPhase (α=2)",
    )
    ax.axhline(e_ref, color="#333333", ls=":", lw=1.4, label=f"ED E₀ = {e_ref:.4f}")
    if 9 in vqe:
        ax.axhline(
            vqe[9],
            color="#8B6FA8",
            ls="-.",
            lw=1.2,
            label=f"VQE best = {vqe[9]:.4f}",
        )
    ax.set_xlabel("Iteration")
    ax.set_ylabel("E₀ (normalized per site)")
    ax.set_title(
        "NQS convergence at N=9 (exact full-summation VMC)",
        fontweight="semibold",
        color="#333",
    )
    ax.legend(fontsize=9)
    ax.set_ylim(min(e_ref, r9.e0, m9.e0) - 0.15, 1.5)
    plt.tight_layout()
    out_conv = FIG / "nqs_vmc_convergence.png"
    plt.savefig(out_conv, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {out_conv.relative_to(REPO)}")

    # --- Figure 2: four-way method comparison (unique vs dmrg_vqe_comparison) ---
    sizes = sorted(rbm_results.keys())
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(len(sizes))
    width = 0.18
    series = [
        ([ed.get(n, np.nan) for n in sizes], "ED", "#EBD8DC"),
        ([dmrg_e.get(n, np.nan) for n in sizes], "DMRG", "#C97B7B"),
        ([rbm_results[n].e0 for n in sizes], "NQS RBM", "#5B8FA8"),
        ([mod_results[n].e0 for n in sizes], "NQS ModPhase", "#8B6FA8"),
        ([vqe.get(n, np.nan) for n in sizes], "VQE best", "#7EB8D4"),
    ]
    offsets = np.linspace(-(len(series) - 1) / 2, (len(series) - 1) / 2, len(series))
    for (vals, label, color), off in zip(series, offsets):
        ax.bar(x + off * width, vals, width, label=label, color=color)
    ax.set_xticks(x, [str(n) for n in sizes])
    ax.set_xlabel("System size N (sites)")
    ax.set_ylabel("E₀ (normalized per site)")
    ax.set_title(
        "ED / DMRG / NQS / VQE on the Kagome strip",
        fontweight="semibold",
        color="#333",
    )
    ax.legend(fontsize=8, ncol=3)
    plt.tight_layout()
    out_cmp = FIG / "nqs_method_comparison.png"
    plt.savefig(out_cmp, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {out_cmp.relative_to(REPO)}")

    # --- Figure 3: relative error vs ED (highlights NQS vs VQE gap) ---
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for n in sizes:
        ref = ed[n]
        methods = {
            "NQS RBM": rbm_results[n].e0,
            "NQS ModPhase": mod_results[n].e0,
            "VQE": vqe.get(n, np.nan),
        }
        xs = list(methods.keys())
        ys = [
            abs(e - ref) / abs(ref) * 100 if e == e else np.nan
            for e in methods.values()
        ]
        ax.plot(xs, ys, "o-", lw=2, ms=8, label=f"N={n}")
    ax.set_ylabel("Relative error vs ED (%)")
    ax.set_title(
        "Variational error: NQS vs VQE",
        fontweight="semibold",
        color="#333",
    )
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, which="both", ls=":", alpha=0.4)
    plt.tight_layout()
    out_err = FIG / "nqs_error_vs_ed.png"
    plt.savefig(out_err, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {out_err.relative_to(REPO)}")

    print("\nSummary (error vs ED):")
    for n in sizes:
        ref = ed[n]
        for label, e in (
            ("RBM", rbm_results[n].e0),
            ("ModPhase", mod_results[n].e0),
            ("VQE", vqe[n]),
        ):
            print(f"  N={n}  {label:8s}  E={e:.8f}  err={abs(e-ref)/abs(ref)*100:.4f}%")


if __name__ == "__main__":
    main()
