"""
Generate figures/qaoa_landscape.png using the same NB04 oracle setup.

Run from repo root:
    python scripts/generate_qaoa_landscape.py

Uses committed data/mp_theta_sh.csv (no MP API key). Does not modify other figures.
"""

from __future__ import annotations

import warnings
from pathlib import Path

from spinq_vqe import qaoa, surrogate, utils

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "figures" / "qaoa_landscape.png"

LAM = 6.0
K = 3
GRID = 40


def main() -> None:
    warnings.filterwarnings("ignore")

    ds = surrogate.load_theta_sh_data()
    sr = surrogate.train_surrogate(
        ds,
        hidden_layer_sizes=(64, 32),
        max_iter=3000,
        random_state=42,
    )
    theta_oracle = surrogate.predict(sr, ds.records)

    print("Sampling p=1 (gamma, beta) landscape...")
    landscape = qaoa.qaoa_landscape_grid(
        theta_oracle,
        k=K,
        lam=LAM,
        p=1,
        n_gamma=GRID,
        n_beta=GRID,
    )
    minima = qaoa.find_landscape_minima(landscape, neighborhood=5, max_minima=4)

    print("Re-running QAOA p=1 with COBYLA trajectory recording...")
    res_p1 = qaoa.run_qaoa(
        theta_oracle,
        k=K,
        p=1,
        lam=LAM,
        n_optimizer_steps=300,
        n_seeds=5,
        step_size=0.3,
        verbose=False,
        record_param_history=True,
    )

    depth_energies: dict[int, float] = {1: res_p1.energy}
    for p in (2, 3):
        print(f"QAOA depth p={p} (cost only)...")
        res = qaoa.run_qaoa(
            theta_oracle,
            k=K,
            p=p,
            lam=LAM,
            n_optimizer_steps=300,
            n_seeds=5,
            step_size=0.3,
            verbose=False,
        )
        depth_energies[p] = res.energy

    OUT.parent.mkdir(parents=True, exist_ok=True)
    utils.plot_qaoa_landscape(
        landscape.gamma,
        landscape.beta,
        landscape.energies,
        cobyla_gamma=float(res_p1.gamma[0]),
        cobyla_beta=float(res_p1.beta[0]),
        param_trajectory=res_p1.param_history,
        local_minima=minima,
        depth_energies=depth_energies,
        save_path=str(OUT),
    )
    print(f"Saved -> {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
