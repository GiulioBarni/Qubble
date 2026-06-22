# Qubble

Numerical code for semiclassical charged vacuum decay at fixed global $U(1)$ charge.

Qubble is the numerical code accompanying *Globally Charged Vacuum Decay* (forthcoming as arXiv:2606.xxxxx).  It constructs fixed-charge Euclidean tunnelling saddles for a homogeneous medium carrying a conserved global $U(1)$ charge, computes the fixed-$Q$ suppression exponent, and uses the Euclidean turning slice as initial data for the subsequent Minkowski evolution.

At finite charge the tunnelling problem is not the ordinary Coleman bounce problem.  The path integral must be projected onto a fixed charge sector, the Euclidean fields obey twisted Euclidean boundary conditions as independent Euclidean fields, and the saddle is naturally formulated in terms of two independent fields with a residual twist $\eta_0$.  The implementation follows this structure directly: it solves a two-dimensional boundary-value problem in $(r,\tau)$, tunes $\eta_0$ to impose the target charge, and then evolves the resulting bubble in real time.

The detailed numerical conventions are described in [`Numerical_implementation.pdf`](Numerical_implementation.pdf).

---

## Repository structure

```text
Qubble/
├── Bubble_finder/              # main fixed-Q bubble-nucleation pipeline
├── Q_ball_finder/              # auxiliary Q-ball / fixed-charge tooling
├── Numerical_implementation.pdf
└── README.md
```

The two code folders have different roles.

`Bubble_finder/` is the main reproduction and analysis pipeline for *Globally Charged Vacuum Decay*.  It contains the scalar potential used in the paper, the 1D reduced bounce solvers, the full 2D fixed-$Q$ Euclidean solver, the residual-twist matching loop, the Euclidean action and charge/energy diagnostics, and the Minkowski continuation after nucleation.

`Q_ball_finder/` is auxiliary.  It contains an earlier-generation implementation for Q-ball-like fixed-charge configurations and Q-ball decay scans.  It is useful for comparison with the Q-ball literature and also provides some shared numerical primitives, such as the staggered half-$\tau$ grid and Newton driver, but it is not the main path for reproducing the bubble-nucleation figures.

---

## Physics problem

The theory is a complex scalar field with a global $U(1)$ symmetry,

$$
\Phi(t,\mathbf{x}) = {\rho(t,\mathbf{x}) \over \sqrt{2}} e^{i\theta(t,\mathbf{x})},
$$

with conserved charge density

$$
q = \rho^2 \dot\theta .
$$

The initial state is a homogeneous rotating charged state,

$$
\Phi_i(t) = {\rho_i \over \sqrt{2}} e^{i\omega_i t},
\qquad
q_i = \omega_i \rho_i^2 .
$$

The decay is treated at fixed total charge.  This requires a charge projection in the Euclidean path integral.  At the saddle, the projection appears as a real exponential twist in Euclidean time.  The Euclidean fields are therefore not a field and its complex conjugate, but two independent fields,

$$
\varphi(\tau,r),\qquad \bar\varphi(\tau,r),
$$

which are recombined into a physical complex field only at the turning slice.  The fixed-$Q$ exponent computed by the code has the form

$$
F_{Q,\beta}
=
S_E[\varphi_b,\bar\varphi_b]
-
S_E[\varphi_i,\bar\varphi_i]
+
\eta_0 Q,
$$

where $\eta_0$ is the residual twist after removing the trivial homogeneous Euclidean rotation.

The real-time evolution starts from the Euclidean turning slice at $\tau=0$.  The modulus is at a turning point, but the phase still carries charge.  This is why the post-nucleation dynamics differs from neutral vacuum decay: charge rearrangement and phase-gradient energy can drive the bubble toward a subluminal terminal velocity.

---

## Main folder: `Bubble_finder/`

This is the main code path.  The folder contains:

| Component | Role |
|---|---|
| `potential_bubble.py` | scalar potential and derivatives used for the bubble-nucleation example. |
| `bounce_1d.py` | reduced O(d)-symmetric 1D bounce solver, used for Coleman references and seed profiles. |
| `ansatz_bubble.py` | construction of O(4), O(3), O(1), and homogeneous seeds on the 2D grid. |
| `bounce2d.py` | main 2D Euclidean fixed-$Q$ solver with twisted boundary conditions and analytic sparse Jacobian. |
| `observables_1d.py`, `observables_2d.py` | charge, energy, and diagnostic observables. |
| `rate_exponent.py` | fixed-charge Euclidean action and tunnelling exponent. |
| `minkowski_evolution_post_tunneling.py` | real-time evolution from the turning slice. |
| `diagnostics_sanity.py` | solver sanity checks and Jacobian tests. |
| `notebooks/` | analysis and figure-production notebooks. |
| `tests/` | pytest tests for the bubble solver. |

See [`Bubble_finder/README.md`](Bubble_finder/README.md) for a more detailed file-by-file description.

---

## Auxiliary folder: `Q_ball_finder/`

This folder is retained for continuity with the Q-ball development path and for cross-checks.  Its boundary conditions are different from the homogeneous-medium bubble problem:

- `Q_ball_finder/`: Q-ball / Q-cloud-type configurations approaching the symmetric vacuum at large radius.
- `Bubble_finder/`: fixed-$Q$ bubbles approaching a homogeneous charged medium at large radius.

The Q-ball folder includes a logistic Q-ball potential, 1D O(d) bounce tools, a 2D Q-ball fixed-charge solver, seed selection, $\beta$ continuation, $\tilde\omega$ scans, and Q-ball observables.

See [`Q_ball_finder/README.md`](Q_ball_finder/README.md) for details.

---

## Installation

The code is pure Python and is intended to be run from the repository root.

A minimal environment is:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy scipy matplotlib jupyter pytest
```

If you run notebooks or scripts directly from the repository root, the local packages should be importable.  If needed, set

```bash
export PYTHONPATH="$PWD:$PYTHONPATH"
```

before launching Jupyter.

---

## Quick checks

Run the fast tests with

```bash
python -m pytest Bubble_finder/tests
python -m pytest Q_ball_finder/tests
```

The full Euclidean and Minkowski production runs are more expensive than the tests.  They should be run through the notebooks or through dedicated scripts after the fast checks pass.

---

## Reproducing the main analysis

The main analysis is organised around the notebooks in `Bubble_finder/notebooks/`.

A typical workflow is:

1. construct the homogeneous charged reference state for a chosen $\omega$;
2. compute the reduced 1D O(d) profiles used as seeds;
3. build the O(4), O(3), O(1), and homogeneous 2D ansatz families;
4. solve the 2D Euclidean fixed-$Q$ boundary-value problem;
5. tune the residual twist $\eta_0$ until the target charge is matched;
6. compute the fixed-$Q$ exponent and the Euclidean energy/charge checks;
7. continue the turning slice to Minkowski time;
8. extract radius, velocity, charge redistribution, wall energy, and terminal-velocity diagnostics;
9. run the post-processing cells that generate the publication figures.

Heavy scans should be cached.  The notebooks are written so that scan cells can save intermediate results and later reload them, avoiding unnecessary reruns.

---

## Data and generated files

Large generated files should not be committed to the public repository.  In particular, avoid committing:

```text
*.npz
*.npy
*.pkl
*.mp4
*.gif
__pycache__/
.ipynb_checkpoints/
data/heavy_scans/
```

Small text caches that are needed for quick reproducibility may be tracked case by case.  Large scan outputs and videos should be regenerated locally or archived externally.

---

## Naming conventions and normalization warnings

There are two common sources of mistakes.

First, the paper uses the physical modulus $\rho$, while parts of the code use the solver-normalized field amplitude $\phi=\rho/\sqrt{2}$.  In particular, whenever a code variable is called `rho0`, check the local file documentation: in some solver contexts it denotes the homogeneous solver amplitude rather than the physical modulus.

Second, the residual twist used in the numerical solver is not the full Euclidean twist.  The homogeneous Euclidean rotation has already been factored out.  The tuned variable is therefore

$$
\eta_0 = \eta_{\rm bounce} - \beta\omega_i .
$$

This is the quantity entering the fixed-charge Legendre term $\eta_0 Q$ in the rotated-frame implementation.

---

## Documentation

- [`Numerical_implementation.pdf`](Numerical_implementation.pdf): self-contained technical notes on the numerical implementation.
- [`Bubble_finder/README.md`](Bubble_finder/README.md): detailed documentation of the main bubble-nucleation code.
- [`Q_ball_finder/README.md`](Q_ball_finder/README.md): detailed documentation of the auxiliary Q-ball code.

---

## Citation

If you use this code, please cite:

Giulio Barni and José R. Espinosa, “Globally Charged Vacuum Decay”, arXiv:2606.xxxxx.

```bibtex
@article{BarniEspinosa2026,
  author       = {Barni, Giulio and Espinosa, Jos{\\'e} R.},
  title        = {Globally Charged Vacuum Decay},
  year         = {2026},
  eprint       = {2606.xxxxx},
  archivePrefix= {arXiv},
  primaryClass = {hep-ph}
}
```

---

## Status

This repository is a research codebase.  The main algorithms are documented and tested at the level needed to reproduce the results of *Globally Charged Vacuum Decay*, but the code is not a general-purpose vacuum-decay package.  When extending it to new potentials or boundary conditions, check charge conservation, Euclidean energy conservation, the $Q\to0$ limit, and the stability of the Minkowski continuation.
