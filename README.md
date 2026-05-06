# Qubble

> **Tunneling decay at fixed charge — research code accompanying the paper**
> *"Tunneling decay at fixed charge: nucleation theory"*
> by G. Barni and J. R. Espinosa.

Qubble (a contraction of "Q-bubble") is the numerical implementation used in
the paper to study semiclassical bubble nucleation from a homogeneous charged
medium at fixed conserved global charge $Q$.

This is a **publication reproducibility companion**, not a general-purpose
finite-charge tunnelling package. The code is tailored to the explicit
scalar model studied in the paper and to the specific numerical methods
described there.

---

## Overview

Qubble implements the fixed $Q$ semiclassical framework developed in the paper.
The decay of a homogeneous charged medium is formulated in a definite charge
sector, and the path integral is projected onto fixed $Q$. After Wick rotation,
charge projection is conjugate to a residual twist $\eta_0$ in Euclidean time,
and the saddle is solved as a real doubled-field problem in the independent
Euclidean fields $\phi$ and $\bar\phi$.

The numerical pipeline computes:

- neutral Coleman O(4) reference bounces (zero-charge limit),
- reduced 1D bounce profiles for d = 1, 3, 4 at fixed frequency $\omega$,
- two-dimensional fixed $Q$ Euclidean saddles in $(r, \tau)$ with twisted
  boundary conditions and a Newton solver with analytic sparse Jacobian,
- the fixed $Q$ decay exponent relative to the homogeneous charged
  reference state in the same charge sector,
- static-energy decomposition and energy/charge diagnostics,
- branch structure from O(4), O(3), and O(1) seeds,
- post-tunnelling Minkowski evolution from the turning slice $\tau=0$,
  including wall velocity and phase-gradient diagnostics,
- publication-ready figures used in the paper.

The implementation verifies the $Q\to 0$ Coleman limit and studies the
finite-$Q$ deformation of the static-energy barrier and of the decay
exponent.

---

## Repository structure

```text
Qubble/
├── Bubble_finder/                # main fixed Q bubble nucleation pipeline
│   ├── __init__.py
│   ├── potential_bubble.py       # scalar potential V(phi) and Omega(phi)
│   ├── bounce_1d.py              # reduced 1D O(d) bounce solver
│   ├── bounce2d.py               # 2D Euclidean (tau, r) Newton solver
│   ├── ansatz_bubble.py          # O(4)/O(3)/O(1) seed construction
│   ├── branching.py              # branch continuation from sphaleron-like roots
│   ├── observables_1d.py         # 1D charge, energy, F_omega
│   ├── observables_2d.py         # 2D charge/energy with tau=0 ghost reconstruction
│   ├── rate_exponent.py          # F^bounce_{Q,beta} (background-subtracted)
│   ├── minkowski_evolution_post_tunneling.py  # real-time evolution from tau=0
│   ├── diagnostics_sanity.py     # Jacobian / background sanity checks
│   ├── clean_analysis_helpers.py # reusable helpers for the main notebook
│   ├── notebooks/
│   │   ├── Bubble_Tunneling_2D.ipynb     # main paper notebook
│   │   └── figures/                       # generated paper figures (PDF/PNG)
│   ├── docs/                              # placeholder for technical PDF
│   └── tests/
│
├── Q_ball_finder/                # auxiliary Q-ball / fixed charge solvers
│   ├── __init__.py
│   ├── potentials.py             # logistic Q-ball potential
│   ├── bounce_solver.py          # general O(d) bounce
│   ├── bounce2d.py               # 2D Q-ball bounce solver
│   ├── ansatz.py, seeds.py, seed_selection.py
│   ├── grid.py, nr_solver.py
│   ├── observables2d.py, qball_observables.py
│   ├── qball_scan.py             # frequency / charge scans
│   ├── beta_scan_continuation.py # Euclidean-time extent continuation
│   ├── rate_exponent.py
│   ├── notebook_utils.py, diagnostics.py
│   ├── notebooks/
│   │   ├── q_ball_explanation.ipynb
│   │   └── q_ball_2D_solution.ipynb
│   ├── data/                     # cached solutions (mostly gitignored, see below)
│   └── tests/
│
├── README.md                     # this file
├── requirements.txt
├── environment.yml
├── CITATION.cff
├── CONTRIBUTING.md
├── LICENSE                       # MIT
└── .gitignore
```

The two scientific subfolders carry their own README files with a more
detailed explanation of conventions, files, and workflows:

- `Bubble_finder/README.md` — main fixed $Q$ pipeline.
- `Q_ball_finder/README.md` — Q-ball / fixed charge tunneling decay.

---

## Scientific background

Standard zero-temperature vacuum decay is described by Coleman's neutral O(4)
bounce. When the initial state is a **homogeneous charged medium** with a
conserved global charge $Q$, the standard Coleman picture has to be modified:

1. The decay must be formulated in a definite charge sector, so the path
   integral is **projected** onto fixed $Q$.
2. The projection inserts a Lagrange multiplier conjugate to $Q$, which after
   Wick rotation becomes a **residual twist $\eta_0$** in Euclidean time.
3. The Euclidean saddle is generically **complex** in the original variables.
   A real numerical formulation is recovered by treating $\phi$ and
   $\bar\phi$ as independent Euclidean fields. Reality of the physical
   continuation is then a derived property of the saddle, not an input.
4. The relevant suppression compares the nontrivial bounce with the
   homogeneous charged reference configuration **in the same charge sector**.

Schematically, the fixed charge decay exponent is

$$
F_{Q,\beta}^{\rm bounce}
= S_E[\phi_b,\bar\phi_b]-S_E[\phi_i,\bar\phi_i]+\eta_0\,Q,
$$

where $\phi_b$ is the nontrivial Euclidean saddle, $\phi_i$ denotes the
homogeneous charged reference, and $\eta_0$ is the residual twist.

The numerical pipeline:

- recovers the neutral Coleman O(4) bounce as $Q \to 0$,
- shows that at finite $Q$ the bounce departs from exact O(4) symmetry,
  the static-energy barrier is lowered, and the decay exponent decreases,
- continues the Euclidean turning slice at $\tau=0$ to real Minkowski time
  and exhibits charge rearrangement, phase-gradient energy near the wall,
  and a subluminal terminal wall velocity even at zero temperature.

A detailed presentation of the formalism is given in the paper. The
self-contained technical description of the numerical implementation will be
provided in `Bubble_finder/docs/numerical_implementation.pdf` (see
`Bubble_finder/docs/README.md`).

---

## Installation

Qubble targets Python ≥ 3.10 (tested with 3.11). The code does not contain
compiled extensions; everything goes through NumPy and SciPy.

### Using `pip` and a virtual environment

```bash
git clone https://github.com/GiulioBarni/Qubble.git
cd Qubble
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

To register the environment as a Jupyter kernel:

```bash
python -m ipykernel install --user --name qubble --display-name "Python (qubble)"
```

### Using `conda`

```bash
git clone https://github.com/GiulioBarni/Qubble.git
cd Qubble
conda env create -f environment.yml
conda activate qubble
```

### Sanity check

```bash
pytest Bubble_finder/tests
pytest Q_ball_finder/tests
```

The test suite is intentionally light. It covers the 2D solver background
exactness, the analytic Jacobian, a few Newton iterations, the 1D bounce
solver on a quartic potential, and the Newton helper.

---

## Quick start

The minimal workflow to reproduce the main paper results is:

1. Install the dependencies (see above).
2. Open the main notebook:
   ```text
   Bubble_finder/notebooks/Bubble_Tunneling_2D.ipynb
   ```
3. Run the **setup / model** cells (Sections 1–2). This sets the scalar
   potential $V(\phi)$ used in the paper and chooses a reference frequency
   $\omega$.
4. Run the **1D neutral and reduced bounce** cells (Sections 3–5). These
   produce the Coleman reference bounces and a 1D scan in $\omega$.
5. Run the **2D solver setup and seeds** (Sections 6–9). This builds the
   default 2D grid and constructs O(4)/O(3)/O(1) seeds.
6. Run the **2D Newton solve and $\eta_0$ scan** (Sections 10–11). This
   produces the fixed $Q$ saddle and computes the decay exponent.
7. Run the **diagnostics** (Sections 12–13): static-energy decomposition,
   parallel analyses for different seeds.
8. Optionally run the **continuation scans** (Sections 14–16) — these
   are the most expensive cells.
9. Run the **post-tunnelling Minkowski evolution** (Section 17) to obtain
   the wall-velocity and phase-gradient diagnostics.

The final plotting cells (the "Plots for paper" subsections) **do not
introduce new physics input**. They reorganise quantities that were already
computed in the analysis cells above into publication-ready figures and
save them to `Bubble_finder/notebooks/figures/`.

---

## Reproducing the paper figures

The paper figures are produced by the main notebook
`Bubble_finder/notebooks/Bubble_Tunneling_2D.ipynb` and are written to
`Bubble_finder/notebooks/figures/`. The auxiliary Q-ball notebooks in
`Q_ball_finder/notebooks/` produce comparison figures and were used during
the development of the formalism; they are not on the main paper figure
critical path but they are kept for cross-checks and continuity with the
Q-ball literature.

The files in `Bubble_finder/notebooks/figures/` fall into four categories:

| Category | Examples |
|---|---|
| **Paper figures** (PDF) | `potential_and_omega_vacua.pdf`, `o4_energy_static.pdf`, `o4_energy_static_Qneq0_vs_Coleman_Q0.pdf`, `o4_red_vs_1d_profiles.pdf`, `o4_interface_and_charge_flow.pdf`, `o4_fancy_full_symmetry_map.pdf`, `decay_rate_vs_beta.pdf`, `minkowski_combined_evolution.pdf`, `minkowski_all_energy_contributions.pdf`, … |
| **Diagnostic plots** (PDF) | per-seed comparisons (`comparison_*.pdf`, `seed_*.pdf`, `solution_*.pdf`), continuation panels (`beta_continuation_*.pdf`), $\omega$-scan panels (`omega_scan_*.pdf`). |
| **Development plots** (PNG) | older PNG snapshots kept for traceability (`bounce_profile.png`, `branch_scan_A.png`, …). |
| **Cached intermediates** | `.npz` files in `Q_ball_finder/data/` (gitignored — see below). |

> Some 2D Newton solves and continuation scans are computationally expensive
> (minutes to tens of minutes per scan point on a workstation). The notebooks
> use cached `.npz` solutions where available; see the *Data and cached
> outputs* section below. Bitwise reproducibility across machines and library
> versions is **not** guaranteed.

---

## Data and cached outputs

The repository ships with a small amount of binary data in `Q_ball_finder/data/`
used by the auxiliary Q-ball pipeline:

- `cloud_scan.txt` — small text file with a frequency scan, **kept in git**.
- `Qball_solution_Q_353.npz`, `ansatz_solution_beta_55.npz` — example cached
  solutions, **gitignored** (regenerable from `Q_ball_finder/notebooks/`).
- `beta_scan_solutions/` — the heavy beta-scan dataset (~140 MB, dozens of
  `.npz` files), **gitignored** and intended for external archiving. 
  This is documented in `Q_ball_finder/README.md`.

The main `Bubble_finder/` pipeline currently does not ship cached `.npz`
solutions; the analysis is rerun from the notebook. If cached solutions are
added in the future, they should also be archived externally and referenced
from the notebooks.

## Tests

The project ships a small `pytest` suite:

```bash
pytest                              # run everything
pytest Bubble_finder/tests          # main 2D solver tests
pytest Q_ball_finder/tests          # auxiliary Q-ball tests
```

Coverage is intentionally narrow: it locks in the background exactness of the
2D solver, the analytic Jacobian against finite differences, a quartic 1D
bounce reference, and the Newton helper. The tests are not a substitute for
running the notebooks end-to-end on the paper figures.

---

## Citation

If you use Qubble, please cite the accompanying paper:

> G. Barni and J. R. Espinosa,
> *"Tunneling decay at fixed charge: nucleation theory"*,
> arXiv:XXXX.XXXXX (2026).

A machine-readable citation is provided in `CITATION.cff`. The arXiv and DOI
fields will be filled in once the preprint and journal references are
finalised.

---

## License

Qubble is released under the MIT License. See [`LICENSE`](LICENSE) for the
full text.

---

## Contact

- **Giulio Barni** — IFT-UAM/CSIC, Madrid.
- mail: giulio.barni@ift.csic.es or giulio.barni95@gmail.com
- For bugs and questions, please open a
  [GitHub issue](https://github.com/GiulioBarni/Qubble/issues).
- See [`CONTRIBUTING.md`](CONTRIBUTING.md) for guidance on filing issues or
  submitting pull requests.
