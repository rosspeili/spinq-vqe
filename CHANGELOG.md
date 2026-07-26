# Changelog

All notable changes to this project can be documented in this file.

The format follows the spirit of [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project uses semantic versioning through the package version in
`pyproject.toml`.

## [Unreleased]

### Changed

- Clarified the NB04 data boundary in `OVERVIEW.md`: Materials Project provides
  structure metadata, θ_SH targets remain curated literature labels, and the
  notebook demonstrates the method rather than claiming material discovery.

## [0.1.2] - 2026-07-03

### Changed

- Rewrote `OVERVIEW.md` as a public-facing research narrative with key results,
  scientific context, roadmap, and cross-repo dependencies (#3).
- Linked `OVERVIEW.md` from `README.md` and `docs/README.md`.
- Removed internal planning language from `docs/notebooks.md` (status badges),
  `docs/physics.md`, `src/spinq_vqe` module docstrings, and notebook headers
  in NB04/NB05 (#3).
- Updated `src/spinq_vqe/__init__.py` module descriptions for `surrogate` and
  `qaoa` to behaviour-focused wording with no internal cluster codes (#4).
- NB04 uses `load_theta_sh_data()` instead of hardcoded mock data (#5).
- `fetch_curated_mp_dataset()` replaces stub `load_mp_data()` — correct MP IDs
  by formula, literature θ_SH (#5).

### Fixed

- NB04: `ax.axhline` placeholder and `greedy["total"]` dict access after
  `classical_greedy` return-type change; regenerated figures and
  `data/qaoa_results.csv` with MP-grounded CSV inputs.

### Added

- Committed `data/mp_theta_sh.csv` — 12 spintronic materials with MP structure
  descriptors and literature θ_SH labels (#5).
- `load_theta_sh_data()` CSV-first loader; `scripts/fetch_mp_theta_sh.py` to
  refresh data via Materials Project API when `MP_API_KEY` is set (#5).
- `.env.example` for optional MP API key; `.env` added to `.gitignore` (#5).

## [0.1.1] - 2026-07-03

### Added

- Citation metadata in `CITATION.cff`.
- Conda-style environment specification in `environment.yml`.
- Initial changelog for release and publication-readiness tracking.
- Pytest unit test suite (`tests/`) covering `kagome`, `ansatz`, `vqe`,
  `entanglement`, `surrogate`, and `qaoa`, with shared session fixtures in
  `conftest.py`.
- Testing guide in `docs/testing.md` (how to run, coverage map, fixtures).
- Notebook 04 (`04_soc_qaoa.ipynb`): SOC QAOA workstream — MLP surrogate on a
  12-material mock dataset, QAOA at depths p = 1/2/3, greedy and simulated-
  annealing baselines; results in `data/qaoa_results.csv` and
  `figures/qaoa_*.png`.

### Changed

- `classical_greedy` in `qaoa.py` now returns `list[int]` (top-k material
  indices) instead of a dict.

### Fixed

- Notebook 04: updated `classical_greedy` call sites after the `list[int]`
  return-type refactor.

## [0.1.0] - 2026-06-01

### Added

- Initial Python package for variational quantum simulation of
  antiferromagnetic Hamiltonians.
- Kagome lattice Hamiltonian utilities, variational ansatz modules, VQE
  runners, entanglement analysis helpers, surrogate modeling, and QAOA tools.
- Research notebooks, generated figures, reference data, documentation, and
  bibliography files.
