# Q-ball finder

Auxiliary fixed-charge / Q-ball solver tooling that accompanies the main
`Bubble_finder/` pipeline.

This folder contains an earlier-generation implementation of the fixed-charge
machinery, applied to Q-ball-like configurations and to Q-ball-decay scans.
It pre-dates the bubble-nucleation pipeline in `Bubble_finder/` and shares
several primitives with it (radial/time grid, Newton driver, packed-field
representation), but uses a different scalar potential and is **not** the
main reproduction pipeline for *Globally Charged Vacuum Decay*.

For the main paper pipeline, see [`Bubble_finder/README.md`](../Bubble_finder/README.md)
(*Globally Charged Vacuum Decay*, forthcoming as arXiv:2606.xxxxx).

---

## Purpose

`Q_ball_finder/` covers:

- the logistic Q-ball scalar potential
  $V(\rho) = -(m^2 v^2 / b)\,\log(e^{-b\rho/v^2} + c)$ used in the original
  Q-ball calculations;
- a general-purpose 1D O(d) bounce solver based on overshoot/undershoot,
  used for both bistable potentials and Q-ball effective potentials;
- a 2D Q-ball / fixed-charge bounce solver in independent fields
  $(y, \bar y)$, formulated on the same half-$\tau$ grid used by
  `Bubble_finder/bounce2d.py`;
- ansatz construction for Q-ball-like configurations, including
  negative-mode-based and plateau-based seeds;
- deterministic seed selection and basin tests over grids of ansatz
  parameters;
- a $\beta$-continuation scan that propagates a converged solution along
  the Euclidean-time extent;
- frequency / $\tilde\omega$ scans matching a target charge between
  metastable and cloud-like solutions;
- observables, decay-exponent computation, and post-processing utilities;
- two notebooks: one explaining the Q-ball setup and matching, and one
  performing the 2D solve and the $\beta$-scan.

Several primitives in this folder (notably `grid.py`, `nr_solver.py`, the
packed-field convention, and the half-$\tau$ staggered grid layout) are
imported by `Bubble_finder/bounce2d.py` when available, with a built-in
fallback if `Q_ball_finder/` is not on the path. Keeping this folder in the
repository therefore also documents those shared primitives.

---

## Scientific role

The paper's homogeneous charged medium can be viewed as related to the
infinite-radius limit of Q-ball-type configurations: a non-trivial Q-ball
profile shrinks to a homogeneous rotating background as the radius
diverges. The material in this folder was developed first, with that
correspondence in mind, and provided several of the building blocks that
were later generalised in `Bubble_finder/` to the **bubble-nucleation
problem**.

The boundary conditions are however different in the two cases:

- **`Q_ball_finder/`**: Q-ball / Q-cloud-type profiles that decay to the
  symmetric vacuum at large $r$.
- **`Bubble_finder/`**: fixed-$Q$ bounces that approach a **spatially
  homogeneous charged configuration** at large $r$.

For this reason, `Q_ball_finder/` is not on the main reproduction path of
the paper figures. It is kept in the repository for cross-checks,
continuity with the Q-ball literature, and as the home of the shared grid
and Newton infrastructure.

---

## File overview

| File | Role |
|---|---|
| `__init__.py` | re-exports the main public symbols of the subpackage. |
| `potentials.py` | logistic Q-ball potential $V(\rho)$ together with its $\chi = \sqrt\rho$ representation, the rotating-frame effective potential, derivatives, and the `LogisticPotentialParams` dataclass. Also includes simple toy potentials (quartic) used by the unit tests. |
| `bounce_solver.py` | general-purpose 1D O(d) bounce solver supporting both bistable and unbounded-on-one-side potentials. Returns the radial profile, $S_E$, $\phi(0)$, false-vacuum location, and termination radius. |
| `bounce2d.py` | the **2D Q-ball / fixed-charge solver** (`QBall2DSettings`, `QBall2DSolver`, `QBall2DSolution`). Uses the same half-$\tau$ staggered grid as `Bubble_finder/bounce2d.py` and the same packed $(y, \bar y)$ representation. Includes an `eta0` outer loop (`scan_eta_to_match_charge`) and the equation-(3.9) charge profile (`compute_charge_profile_eq39`). |
| `grid.py` | grid construction (`build_grid`, `RadialTimeGrid`) and the field conversions $y \leftrightarrow \phi$, $\bar y \leftrightarrow \bar\phi$ via the rotation $y = r\,e^{-\omega\tau}\phi$. **Reused by `Bubble_finder/bounce2d.py`.** |
| `nr_solver.py` | the Newton driver (`newton_solve`, `NewtonResult`, `NewtonConvergenceError`) with sparse-aware linear solves and optional damping/line search. **Reused by `Bubble_finder/bounce2d.py`.** |
| `ansatz.py` | construction of seeds for the 2D solver: negative-mode-based ansatz (`build_negative_mode_ansatz`), Q-ball-like plateau ansatz (`build_qball_like_plateau_ansatz`), Q-ball escape ansatz, and the `AnsatzResult` data class. |
| `seeds.py` | additional seed-construction utilities. |
| `seed_selection.py` | deterministic seed selection by minimising $\lVert F(x_0) \rVert$ over grids of ansatz parameters. Returns the best seed plus a diagnostic table. |
| `profiles.py` | radial Q-ball profile $\chi(r)$ from the 1D bounce solver, plus computation of unstable modes (`UnstableMode`) used to seed escape directions. |
| `observables2d.py` | 2D charge $Q$ (Eq. 3.9 convention) and energy $E$ on the half-$\tau$ grid. |
| `qball_observables.py` | 1D Q-ball observables: charge, energy, dimensionless $g^2 Q$. |
| `qball_scan.py` | precomputed scan in $\tilde\omega$ matching a target charge between metastable Q-balls and the corresponding Q-clouds. Caches results to `data/cloud_scan.txt`. |
| `beta_scan_continuation.py` | helpers for continuing a converged 2D solution along $\beta$, including $N_\tau$ rescaling and warm-starting. |
| `notebook_utils.py` | resampling, warm-starting, and plot helpers for the notebooks. |
| `diagnostics.py` | basic sanity diagnostics. |
| `rate_exponent.py` | computation of the Q-ball-decay exponent / rate. |
| `notebooks/q_ball_explanation.ipynb` | explanatory notebook: model parameters, metastable Q-ball, matching Q-cloud, effective potentials, profile comparisons, 2D ansatz preparation. |
| `notebooks/q_ball_2D_solution.ipynb` | 2D solve and $\beta$-scan notebook with the production runs. |
| `notebooks/q_ball_explanation.ipynb.backup` | retained backup snapshot of the explanation notebook. |
| `data/` | small cached data and pre-computed solutions (see *Data* below). |
| `tests/test_bounce_solver.py` | regression tests against the original notebook values for the 1D bounce solver (quartic and logistic Q-ball potentials). |
| `tests/test_nr_solver.py` | unit test of the Newton driver on a simple quadratic. |
| `NOTEBOOK_BETA_SCAN_SNIPPETS.md`, `NOTEBOOK_SCAN_WORKFLOW_STEPS_ABC.md` | documentation snippets describing how the $\beta$-scan and the $\eta_0$-matching workflows were assembled in the notebooks. |

---

## Workflow

A typical workflow with this folder, mirroring the structure of the two
notebooks, is:

1. **Select model parameters** with `LogisticPotentialParams(m, v, b)` and
   choose a Q-ball frequency $\omega$ on the metastable branch.
2. **Build or load initial seeds** — either a negative-mode ansatz built
   from `compute_unstable_mode` and `build_negative_mode_ansatz`, or a
   Q-ball-like plateau / escape ansatz.
3. **Solve the radial / 1D Q-ball profile** with
   `bounce_solver.solve_bounce` and the rotating-frame effective potential
   from `effective_qball_potential`.
4. **Build the 2D grid** via `build_grid(Nr, Ntau, Lr, beta)`.
5. **Run the 2D solver** with `QBall2DSolver(...).solve(x0)` from a chosen
   seed, optionally followed by the $\eta_0$ outer loop
   (`scan_eta_to_match_charge`) to match a target charge.
6. **Scan in $\beta$ or $\tilde\omega$** using
   `beta_scan_continuation` / `qball_scan` if needed.
7. **Compute observables** on the converged solution
   (`observables2d.compute_charge`, `compute_energy`, the Eq.-(3.9) charge
   profile, `qball_observables.compute_dimensionless_charge`).
8. **Run the diagnostics** in `diagnostics.py` to check basic sanity.
9. **Save / load cached solutions** as `.npz` files in `data/`.

---

## Data

The `data/` directory contains:

| File | Size | Tracked in git? | Notes |
|---|---|---|---|
| `cloud_scan.txt` | ~4 KB | yes | small text file with the precomputed $\tilde\omega$ scan used by `qball_scan.py`. Regenerable with `precompute_cloud_scan(...)`. |
| `Qball_solution_Q_353.npz` | ~3 MB | no (gitignored via `*.npz`) | example 2D Q-ball solution at $Q \approx 353$, used by the notebooks for warm-starting. |
| `ansatz_solution_beta_55.npz` | ~1 MB | no | example seed solution at $\beta = 55$. |
| `beta_scan_solutions/` | ~140 MB | **no — heavy dataset** | dozens of converged 2D solutions along the $\beta$-scan. Not tracked in git. **Should be archived externally (Zenodo) before public release**, with the resulting DOI added to this README. |

The heavy `beta_scan_solutions/` dataset is regenerable from
`q_ball_2D_solution.ipynb` (the $\beta$-continuation cells), but doing so is
expensive: tens of minutes per scan point on a workstation. For practical
use, downloading the cached dataset from the external archive is
recommended.

---

## Tests

```bash
pytest Q_ball_finder/tests
```

The suite contains:

- `test_quartic_bounce_matches_notebook_values` — locks in numerical
  agreement of `bounce_solver.solve_bounce` with the values originally
  produced in the development notebook for both a quartic potential and a
  logistic Q-ball case.
- `test_newton_solver_simple_quadratic` — solving $x^2 = 2$ with
  `nr_solver.newton_solve`.

These tests are fast and self-contained.

---

## Relation to Qubble and the paper

This folder is **auxiliary**: it was developed before the bubble-nucleation pipeline in
`Bubble_finder/`, it operates on a different scalar potential, and it
implements Q-ball-type boundary conditions rather than the bubble-on-medium
boundary conditions used in the paper. The shared primitives (`grid.py`,
`nr_solver.py`, the half-$\tau$ staggered grid layout, the packed
$(y, \bar y)$ representation) are however imported by `Bubble_finder/`
through soft imports, so this folder is part of the runtime dependency
graph of the main pipeline.

If you are reproducing the figures of *Globally Charged Vacuum Decay*
(arXiv:2606.xxxxx), the entry point is
`Bubble_finder/notebooks/Bubble_Tunneling_2D_public.ipynb`. The notebooks in
this folder are useful for cross-checks against the Q-ball literature and for
exploring the shared infrastructure, but they are not the paper's
reproduction critical path.
