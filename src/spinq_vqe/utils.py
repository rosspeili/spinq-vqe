"""
utils.py
--------
Plotting helpers and Pauli string utilities for spinq-vqe.

Functions
---------
plot_kagome_graph        : Visualize the Kagome lattice with sublattice colors
plot_energy_convergence  : Plot VQE energy history for HEA vs MERA
plot_entanglement_profile: Plot S_vN vs bipartition size
plot_mutual_info_matrix  : Heatmap of sublattice mutual information
plot_gradient_variance   : Barren plateau diagnostic plot
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# ---------------------------------------------------------------------------
# Soft pastel palette
# ---------------------------------------------------------------------------

# Sublattice colors: muted lavender, warm peach, soft sage
SUBLATTICE_COLORS = {0: "#B8B8E8", 1: "#F5C9A0", 2: "#A8D8B0"}
SUBLATTICE_LABELS = {0: "A", 1: "B", 2: "C"}

# Ansatz trace colors: dusty blue and muted coral
ANSATZ_COLORS = {"hea": "#7EB8D4", "mera": "#E8A598"}

# Accent / reference line
REF_COLOR = "#B0B0B0"

# Heatmap colormap (soft warm ramp)
HEATMAP_CMAP = "YlOrBr"

# Plot style — light, clean, publication-ready
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.facecolor": "#FAFAFA",
        "figure.facecolor": "#FFFFFF",
        "axes.edgecolor": "#CCCCCC",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.grid": True,
        "grid.color": "#EBEBEB",
        "grid.linewidth": 0.7,
        "xtick.color": "#666666",
        "ytick.color": "#666666",
        "axes.labelcolor": "#444444",
        "text.color": "#333333",
        "figure.dpi": 120,
    }
)


# ---------------------------------------------------------------------------
# Lattice visualization
# ---------------------------------------------------------------------------


def plot_kagome_graph(
    G: nx.Graph,
    title: str = "Kagome Lattice",
    figsize: tuple = (8, 4),
    save_path: str | None = None,
) -> plt.Figure:
    """
    Draw the Kagome lattice graph with sublattice coloring.

    Parameters
    ----------
    G : nx.Graph
        Kagome graph from :func:`spinq_vqe.kagome.kagome_graph`.
    title : str
    figsize : tuple
    save_path : str or None
        If given, save the figure to this path.

    Returns
    -------
    plt.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Layout: position sites along a horizontal strip
    n_sites = G.number_of_nodes()
    n_cells = n_sites // 3
    pos = {}
    for k in range(n_cells):
        pos[3 * k] = (3 * k, 0.0)
        pos[3 * k + 1] = (3 * k + 1.0, 0.0)
        pos[3 * k + 2] = (3 * k + 0.5, 0.866)  # equilateral triangle

    node_colors = [SUBLATTICE_COLORS[G.nodes[n]["sublattice"]] for n in G.nodes]
    node_labels = {
        n: f"{n}\n({SUBLATTICE_LABELS[G.nodes[n]['sublattice']]})" for n in G.nodes
    }

    nx.draw_networkx(
        G,
        pos=pos,
        ax=ax,
        node_color=node_colors,
        node_size=650,
        labels=node_labels,
        font_size=8,
        font_color="#444444",
        font_weight="semibold",
        edge_color="#D0D0D0",
        width=1.8,
    )

    # Legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(
            facecolor=SUBLATTICE_COLORS[i],
            edgecolor="#AAAAAA",
            label=f"Sublattice {SUBLATTICE_LABELS[i]}",
        )
        for i in range(3)
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper right",
        framealpha=0.9,
        edgecolor="#DDDDDD",
        fontsize=10,
    )
    ax.set_title(title, fontsize=13, fontweight="semibold", color="#333333")
    ax.set_facecolor("#FAFAFA")
    ax.axis("off")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig


# ---------------------------------------------------------------------------
# VQE convergence
# ---------------------------------------------------------------------------


def plot_energy_convergence(
    results: dict,
    ed_energy: float | None = None,
    title: str = "VQE Energy Convergence",
    figsize: tuple = (9, 4),
    save_path: str | None = None,
) -> plt.Figure:
    """
    Plot energy histories for HEA and MERA VQE runs.

    Parameters
    ----------
    results : dict
        {"hea": VQEResult, "mera": VQEResult} from :func:`spinq_vqe.vqe.compare_ansatze`.
    ed_energy : float or None
        Exact diagonalization energy (horizontal reference line).
    title, figsize, save_path : as usual.

    Returns
    -------
    plt.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    labels = {"hea": "HEA", "mera": "MERA (simplified)"}

    for name, result in results.items():
        ax.plot(
            result.energy_history,
            color=ANSATZ_COLORS[name],
            label=f"{labels[name]}  (final: {result.energy:.5f})",
            linewidth=2.0,
            alpha=0.9,
        )

    if ed_energy is not None:
        ax.axhline(
            ed_energy,
            color=REF_COLOR,
            linestyle="--",
            linewidth=1.4,
            label=f"ED exact: {ed_energy:.5f}",
        )

    ax.set_xlabel("Optimizer step", color="#555555")
    ax.set_ylabel("Energy (normalized)", color="#555555")
    ax.set_title(title, fontsize=13, fontweight="semibold", color="#333333")
    ax.legend(framealpha=0.9, edgecolor="#DDDDDD", fontsize=10)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig


# ---------------------------------------------------------------------------
# Entanglement entropy profile
# ---------------------------------------------------------------------------


def plot_entanglement_profile(
    profile: dict,
    ansatz_label: str = "",
    figsize: tuple = (7, 4),
    save_path: str | None = None,
) -> plt.Figure:
    """
    Plot von Neumann entropy vs subsystem size.

    Parameters
    ----------
    profile : dict
        Output of :func:`spinq_vqe.entanglement.entanglement_profile`.
    ansatz_label : str
    figsize, save_path : as usual.
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(
        profile["subsystem_sizes"],
        profile["entropies"],
        "o-",
        color="#9BB8D4",  # soft dusty blue
        linewidth=2.0,
        markersize=7,
        markerfacecolor="#D4E6F1",
        markeredgecolor="#9BB8D4",
        markeredgewidth=1.2,
        label=ansatz_label or "VQE",
    )

    ax.set_xlabel("Subsystem size |A|", color="#555555")
    ax.set_ylabel("$S_{\\mathrm{vN}}(\\rho_A)$ [bits]", color="#555555")
    ax.set_title(
        "Entanglement Entropy Profile",
        fontsize=13,
        fontweight="semibold",
        color="#333333",
    )
    if ansatz_label:
        ax.legend(framealpha=0.9, edgecolor="#DDDDDD", fontsize=10)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig


# ---------------------------------------------------------------------------
# Sublattice mutual information heatmap
# ---------------------------------------------------------------------------


def plot_mutual_info_matrix(
    matrix: np.ndarray,
    figsize: tuple = (5, 4),
    save_path: str | None = None,
) -> plt.Figure:
    """
    Heatmap of the 3×3 sublattice mutual information matrix.

    Parameters
    ----------
    matrix : np.ndarray, shape (3, 3)
        Output of :func:`spinq_vqe.entanglement.sublattice_mutual_info_matrix`.
    """
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(matrix, cmap=HEATMAP_CMAP, vmin=0, alpha=0.85)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("I(A:B) [bits]", color="#555555", fontsize=10)
    cbar.ax.yaxis.set_tick_params(color="#888888")

    sl_labels = ["A", "B", "C"]
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    ax.set_xticklabels(sl_labels, color="#555555")
    ax.set_yticklabels(sl_labels, color="#555555")

    vmax = matrix.max() if matrix.max() > 0 else 1.0
    for i in range(3):
        for j in range(3):
            ax.text(
                j,
                i,
                f"{matrix[i, j]:.3f}",
                ha="center",
                va="center",
                fontsize=10,
                color="#555555" if matrix[i, j] < vmax * 0.65 else "#F5F0EB",
            )

    ax.set_title(
        "Sublattice Mutual Information I(A:B)",
        fontsize=12,
        fontweight="semibold",
        color="#333333",
    )
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig


# ---------------------------------------------------------------------------
# Barren plateau diagnostic
# ---------------------------------------------------------------------------


def plot_gradient_variance(
    results: dict,
    figsize: tuple = (9, 4),
    save_path: str | None = None,
) -> plt.Figure:
    """
    Plot gradient variance over training steps for HEA vs MERA.
    Vanishing gradient variance = barren plateau signature.
    """
    fig, ax = plt.subplots(figsize=figsize)
    labels = {"hea": "HEA", "mera": "MERA (simplified)"}

    for name, result in results.items():
        ax.semilogy(
            result.gradient_variance_history,
            color=ANSATZ_COLORS[name],
            label=labels[name],
            linewidth=2.0,
            alpha=0.9,
        )

    ax.set_xlabel("Optimizer step", color="#555555")
    ax.set_ylabel("Gradient variance (log scale)", color="#555555")
    ax.set_title(
        "Barren Plateau Diagnostic: Gradient Variance",
        fontsize=13,
        fontweight="semibold",
        color="#333333",
    )
    ax.legend(framealpha=0.9, edgecolor="#DDDDDD", fontsize=10)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig


# ---------------------------------------------------------------------------
# QAOA landscape (NB04 diagnostic)
# ---------------------------------------------------------------------------


def _grid_edges(centers: np.ndarray) -> np.ndarray:
    """Cell boundaries for a uniformly sampled 1-D angle grid."""
    centers = np.asarray(centers, dtype=float)
    if len(centers) == 1:
        return np.array([centers[0] - 0.5, centers[0] + 0.5])
    diffs = np.diff(centers)
    edges = np.empty(len(centers) + 1)
    edges[1:-1] = centers[:-1] + diffs / 2
    edges[0] = centers[0] - diffs[0] / 2
    edges[-1] = centers[-1] + diffs[-1] / 2
    return edges


def _wrap_qaoa_angles(
    gamma: np.ndarray,
    beta: np.ndarray,
    *,
    gamma_period: float = 2 * np.pi,
    beta_period: float = np.pi,
) -> tuple[np.ndarray, np.ndarray]:
    """Map QAOA angles into the standard landscape window for plotting."""
    return np.mod(gamma, gamma_period), np.mod(beta, beta_period)


def _plot_wrapped_qaoa_path(
    ax: plt.Axes,
    gamma: np.ndarray,
    beta: np.ndarray,
    *,
    color: str = "#FFFFFF",
    lw: float = 1.2,
    alpha: float = 0.75,
    label: str = "COBYLA path",
    gamma_period: float = 2 * np.pi,
    beta_period: float = np.pi,
) -> None:
    """Plot a COBYLA trajectory without spurious lines from angle wrapping."""
    g, b = _wrap_qaoa_angles(gamma, beta, gamma_period=gamma_period, beta_period=beta_period)
    if len(g) < 2:
        ax.scatter(b, g, color=color, s=12, alpha=alpha, label=label, zorder=5)
        return

    segments: list[np.ndarray] = []
    current = np.column_stack([b[:1], g[:1]])
    for i in range(1, len(g)):
        if abs(g[i] - g[i - 1]) > gamma_period / 2 or abs(b[i] - b[i - 1]) > beta_period / 2:
            segments.append(current)
            current = np.column_stack([b[i : i + 1], g[i : i + 1]])
        else:
            current = np.vstack([current, [b[i], g[i]]])
    segments.append(current)

    for idx, seg in enumerate(segments):
        ax.plot(
            seg[:, 0],
            seg[:, 1],
            color=color,
            lw=lw,
            alpha=alpha,
            label=label if idx == 0 else "_nolegend_",
            zorder=5,
        )


def plot_qaoa_landscape(
    gamma: np.ndarray,
    beta: np.ndarray,
    energies: np.ndarray,
    *,
    cobyla_gamma: float | None = None,
    cobyla_beta: float | None = None,
    param_trajectory: list[np.ndarray] | None = None,
    local_minima: list[tuple[float, float, float]] | None = None,
    depth_theta_sh: dict[int, float] | None = None,
    classical_theta_sh: float | None = None,
    figsize: tuple[float, float] = (12.0, 4.8),
    save_path: str | None = None,
) -> plt.Figure:
    """
    Two-panel QAOA diagnostic: p=1 (γ, β) cost landscape + depth sensitivity.

    Parameters
    ----------
    gamma, beta : np.ndarray
        1-D angle grids.
    energies : np.ndarray, shape (len(gamma), len(beta))
        Sampled QAOA cost values.
    cobyla_gamma, cobyla_beta : float, optional
        Best COBYLA optimum to mark on the landscape.
    param_trajectory : list of ndarray, optional
        COBYLA parameter vectors (p=1: each shape ``(2,)``).
    local_minima : list of (γ, β, E), optional
        Coarse local minima from ``find_landscape_minima``.
    depth_theta_sh : dict[int, float], optional
        Total selected θ_SH vs QAOA depth ``p`` (higher = better ranking).
    classical_theta_sh : float, optional
        Greedy baseline total θ_SH (horizontal reference line).
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    ax = axes[0]
    beta_edges = _grid_edges(np.asarray(beta, dtype=float))
    gamma_edges = _grid_edges(np.asarray(gamma, dtype=float))
    bb, gg = np.meshgrid(beta_edges, gamma_edges)
    mesh = ax.pcolormesh(
        bb,
        gg,
        energies,
        cmap="viridis",
        shading="flat",
        alpha=0.95,
        rasterized=True,
    )
    cbar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("QAOA cost ⟨H_C⟩", color="#555555", fontsize=10)

    if local_minima:
        mins = np.array(local_minima)
        ax.scatter(
            mins[:, 1],
            mins[:, 0],
            s=55,
            facecolors="none",
            edgecolors="#F5C9A0",
            linewidths=1.4,
            label="local minima",
            zorder=4,
        )

    if param_trajectory:
        traj = np.asarray(param_trajectory)
        if traj.ndim == 2 and traj.shape[1] >= 2:
            _plot_wrapped_qaoa_path(ax, traj[:, 0], traj[:, 1])

    if cobyla_gamma is not None and cobyla_beta is not None:
        cg, cb = _wrap_qaoa_angles(
            np.asarray([cobyla_gamma]),
            np.asarray([cobyla_beta]),
        )
        ax.scatter(
            cb,
            cg,
            s=90,
            color="#E8A598",
            edgecolors="white",
            linewidths=1.0,
            zorder=6,
            label="COBYLA best",
        )

    ax.set_xlabel("β", color="#555555")
    ax.set_ylabel("γ", color="#555555")
    ax.set_title(
        "QAOA p=1 cost landscape",
        fontsize=12,
        fontweight="semibold",
        color="#333333",
    )
    ax.legend(fontsize=8, framealpha=0.9, loc="upper right")

    ax2 = axes[1]
    if depth_theta_sh:
        depths = sorted(depth_theta_sh)
        totals = [depth_theta_sh[p] for p in depths]
        bars = ax2.bar(
            [str(p) for p in depths],
            totals,
            color=ANSATZ_COLORS["hea"],
            alpha=0.85,
            edgecolor="white",
        )
        ax2.bar_label(bars, fmt="%.2f", fontsize=9, color="#444444")
        ymin = min(totals + ([classical_theta_sh] if classical_theta_sh is not None else []))
        ymax = max(totals + ([classical_theta_sh] if classical_theta_sh is not None else []))
        ax2.set_ylim(bottom=min(0.0, ymin) - 0.35, top=ymax * 1.08 + 0.05)
        if classical_theta_sh is not None:
            ax2.axhline(
                classical_theta_sh,
                color="#C7E4CA",
                linestyle="--",
                linewidth=1.8,
                label=f"greedy ({classical_theta_sh:.2f})",
            )
            ax2.legend(fontsize=8, framealpha=0.9, loc="lower right")
        ax2.set_xlabel("QAOA depth p")
        ax2.set_ylabel("Total θ_SH (selected k)")
        ax2.set_title(
            "Material ranking vs depth (higher is better)",
            fontsize=12,
            fontweight="semibold",
            color="#333333",
        )
    else:
        ax2.axis("off")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig
