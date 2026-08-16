"""
nqs.py
------
Neural Quantum State (NQS) baselines for Kagome strips via NetKet.

Builds the same normalized Heisenberg Hamiltonian as
:func:`kagome.heisenberg_kagome_hamiltonian` (graph edges + J/(4N), D/4
identity shift), then optimizes neural wavefunctions with exact
full-summation VMC (small N) or sampled VMC (larger N).

Default models use **complex** parameters: real RBMs cannot represent the
frustrated Kagome sign structure and plateau far above E₀.

References
----------
- Carleo & Troyer (2017) Science 355 — Neural Quantum States
- Đurić et al. (2025) PRX 15, 011047 — GCNN NQS on Kagome Heisenberg
- Vieijra et al. (2020) PRL 124, 097201 — symmetry-adapted RBM
"""

from __future__ import annotations

import csv
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from spinq_vqe.kagome import (
    B_DEFAULT,
    D_MN3SN_MEV,
    J_MN3SN_MEV,
    heisenberg_kagome_hamiltonian,
    kagome_graph,
)

try:
    import netket as nk

    NETKET_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    nk = None  # type: ignore
    NETKET_AVAILABLE = False

ModelName = Literal["rbm", "rbm_modphase", "rbm_symm", "gcnn"]
BackendName = Literal["exact", "sampled", "auto"]


@dataclass
class NQSResult:
    """One NQS optimization run on a Kagome strip."""

    n_cells: int
    n_sites: int
    model: str
    e0: float
    e0_err: float
    n_iter: int
    n_samples: int
    backend: str
    alpha: float | None = None
    energy_history: list[float] = field(default_factory=list)
    err_history: list[float] = field(default_factory=list)


def require_netket() -> None:
    if not NETKET_AVAILABLE:
        raise ImportError(
            "netket is required for NQS. Install with: pip install -e '.[nqs]'"
        )


def build_kagome_netket_hamiltonian(
    n_cells: int,
    *,
    J: float = J_MN3SN_MEV,
    D: float = D_MN3SN_MEV,
):
    """
    NetKet LocalOperator matching PennyLane / TeNPy strip Hamiltonian.

    Uses one XX+YY+ZZ term per ``kagome_graph`` edge at ``J/(4 N)`` and a
    global identity shift ``D/4`` (equivalent to per-site ``D/(4 N)`` identities).
    """
    require_netket()
    graph = kagome_graph(n_cells)
    n_sites = graph.number_of_nodes()
    hi = nk.hilbert.Spin(s=0.5, N=n_sites)
    j_eff = J / (4.0 * n_sites)
    terms: list[Any] = []
    for i, j in sorted(graph.edges()):
        terms.append(
            j_eff
            * (
                nk.operator.spin.sigmax(hi, i)
                @ nk.operator.spin.sigmax(hi, j)
            )
        )
        terms.append(
            j_eff
            * (
                nk.operator.spin.sigmay(hi, i)
                @ nk.operator.spin.sigmay(hi, j)
            )
        )
        terms.append(
            j_eff
            * (
                nk.operator.spin.sigmaz(hi, i)
                @ nk.operator.spin.sigmaz(hi, j)
            )
        )
    hamiltonian = sum(terms[1:], terms[0])
    if abs(D) > 1e-12:
        # n_sites * (D/(4 n_sites)) * I = (D/4) * I
        hamiltonian = hamiltonian + (D / 4.0) * nk.operator.spin.identity(hi)
    return hi, hamiltonian, graph


def validate_hamiltonian_against_pennylane(
    n_cells: int,
    *,
    J: float = J_MN3SN_MEV,
    D: float = D_MN3SN_MEV,
    B: float = B_DEFAULT,
    atol: float = 1e-10,
) -> float:
    """Return max |H_netket - H_pennylane|; raise if above ``atol``."""
    require_netket()
    import pennylane as qp

    hi, hamiltonian, graph = build_kagome_netket_hamiltonian(n_cells, J=J, D=D)
    if abs(B) > 1e-12:
        raise NotImplementedError("NQS Hamiltonian validation currently assumes B=0")
    n_sites = graph.number_of_nodes()
    h_nk = np.asarray(hamiltonian.to_dense())
    h_pl = np.array(
        qp.matrix(
            heisenberg_kagome_hamiltonian(graph, J=J, D=D, B=B),
            wire_order=list(range(n_sites)),
        )
    )
    max_diff = float(np.max(np.abs(h_pl - h_nk)))
    if max_diff > atol:
        raise AssertionError(
            f"NetKet/PennyLane Hamiltonian mismatch at N={n_sites}: "
            f"max diff {max_diff:.3e}"
        )
    return max_diff


def _netket_graph(graph):
    return nk.graph.Graph(
        edges=list(graph.edges()), n_nodes=graph.number_of_nodes()
    )


def _make_model(
    model: ModelName,
    graph,
    *,
    alpha: float,
    features: int,
    complex_params: bool,
):
    require_netket()
    dtype = np.complex128 if complex_params else np.float64
    if model == "rbm":
        return nk.models.RBM(alpha=alpha, param_dtype=dtype)
    if model == "rbm_modphase":
        # Amplitude + phase factorization; amplitude uses real params.
        return nk.models.RBMModPhase(alpha=alpha, param_dtype=np.float64)
    nk_graph = _netket_graph(graph)
    if model == "rbm_symm":
        return nk.models.RBMSymm(
            symmetries=nk_graph,
            alpha=alpha,
            param_dtype=dtype,
        )
    if model == "gcnn":
        # Group-CNN over the automorphism group of the Kagome *strip* graph
        # (not a 2D Kagome point-group GCNN from the literature).
        return nk.models.GCNN(
            symmetries=nk_graph.automorphisms(),
            layers=2,
            features=features,
            param_dtype=dtype,
        )
    raise ValueError(f"Unknown NQS model: {model}")


def _resolve_backend(n_sites: int, backend: BackendName) -> str:
    if backend == "auto":
        # Exact full summation is practical through N=12 (dim 4096).
        return "exact" if n_sites <= 12 else "sampled"
    return backend


def _energy_arrays(log) -> tuple[np.ndarray, np.ndarray]:
    """Extract energy mean and MC error from a NetKet RuntimeLog."""
    hist = log.data["Energy"]
    energy = np.asarray(hist["Mean"]).real
    errs = np.zeros_like(energy)
    # History supports key access (Mean/Variance/Sigma/...); no .dtype field.
    try:
        errs = np.asarray(hist["Sigma"]).real
    except Exception:
        try:
            errs = np.asarray(hist["Error"]).real
        except Exception:
            pass
    return energy, errs


def run_nqs(
    n_cells: int,
    *,
    model: ModelName = "rbm",
    J: float = J_MN3SN_MEV,
    D: float = D_MN3SN_MEV,
    n_iter: int = 400,
    n_samples: int = 2000,
    n_chains: int = 32,
    learning_rate: float = 0.01,
    diag_shift: float = 0.01,
    alpha: float = 2.0,
    features: int = 4,
    seed: int = 42,
    backend: BackendName = "auto",
    complex_params: bool = True,
    optimizer: Literal["adam", "sgd"] = "adam",
    show_progress: bool = False,
) -> NQSResult:
    """
    Optimize an NQS wavefunction for a Kagome strip.

    Returns normalized per-site energy compatible with ED / DMRG / VQE CSVs.
    For ``backend='exact'`` (default when N≤12), expectations use full Hilbert
    summation — no Monte Carlo noise. Complex RBM is required for <5% accuracy
    on this frustrated strip.
    """
    require_netket()
    hi, hamiltonian, graph = build_kagome_netket_hamiltonian(n_cells, J=J, D=D)
    n_sites = graph.number_of_nodes()
    resolved = _resolve_backend(n_sites, backend)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        nk_model = _make_model(
            model,
            graph,
            alpha=alpha,
            features=features,
            complex_params=complex_params,
        )
        if resolved == "exact":
            vstate = nk.vqs.FullSumState(hi, nk_model, seed=seed)
            samples_used = 2**n_sites
        else:
            sampler = nk.sampler.MetropolisLocal(hi, n_chains=n_chains)
            vstate = nk.vqs.MCState(
                sampler, nk_model, n_samples=n_samples, seed=seed
            )
            samples_used = n_samples

        if optimizer == "adam":
            opt = nk.optimizer.Adam(learning_rate=learning_rate)
        else:
            opt = nk.optimizer.Sgd(learning_rate=learning_rate)
        sr = nk.optimizer.SR(diag_shift=diag_shift)
        driver = nk.driver.VMC(
            hamiltonian,
            opt,
            variational_state=vstate,
            preconditioner=sr,
        )
        log = nk.logging.RuntimeLog()
        driver.run(n_iter=n_iter, out=log, show_progress=show_progress)

    energy, errs = _energy_arrays(log)
    # Trailing mean is more stable than a single noisy sample for VMC;
    # for exact backend the curve is smooth and final ≈ min.
    window = max(10, n_iter // 20)
    tail = energy[-window:]
    best_e = float(np.min(tail))
    best_idx = int(np.argmin(tail))
    best_err = float(errs[-window:][best_idx]) if len(errs) else 0.0

    return NQSResult(
        n_cells=n_cells,
        n_sites=n_sites,
        model=model,
        e0=best_e,
        e0_err=best_err,
        n_iter=n_iter,
        n_samples=samples_used,
        backend=resolved,
        alpha=alpha if model.startswith("rbm") else None,
        energy_history=[float(x) for x in energy],
        err_history=[float(x) for x in errs] if len(errs) else [],
    )


def save_method_comparison_csv(
    rows: list[dict[str, Any]],
    path: Path | str,
) -> Path:
    """Write ``data/method_comparison.csv`` (new append-only artifact)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "n_sites",
        "n_cells",
        "E_ED",
        "E_DMRG",
        "E_NQS_RBM",
        "E_NQS_RBM_err",
        "E_NQS_MODPHASE",
        "E_NQS_MODPHASE_err",
        "E_VQE",
        "NQS_RBM_error_vs_ED_pct",
        "NQS_MODPHASE_error_vs_ED_pct",
        "VQE_error_vs_ED_pct",
        "NQS_backend",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    return path


def load_method_comparison_csv(path: Path | str) -> list[dict[str, str]]:
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
