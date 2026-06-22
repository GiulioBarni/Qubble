# `Bubble_finder/`

Main reproduction and analysis pipeline for *Globally Charged Vacuum Decay*
(forthcoming as arXiv:2606.xxxxx). This folder contains the code that constructs
the fixed-$Q$ Euclidean saddle at fixed global $U(1)$ charge, computes the
corresponding semiclassical exponent, and continues the turning slice to real
time.

For the project-level overview, installation instructions, and notebook order,
see the parent README:

```text
../README.md
```

For the derivation of the numerical equations implemented here, see the
technical notes placed next to this folder:

```text
../Numerical_implementation.pdf
```

Those notes are the authoritative reference for the discretized equations,
rotated variables, ghost boundary conditions, Newton solve, fixed-charge outer
loop, ansatz construction, and Minkowski continuation. This README is only the
code map for `Bubble_finder/`.

---

## What this folder does

The physical problem is the semiclassical decay of a homogeneous charged medium
carrying a conserved global charge $Q$. Unlike ordinary Coleman tunnelling, the
Euclidean path integral is projected onto a fixed-charge sector. This introduces
a twist conjugate to $Q$, and the saddle is naturally formulated in terms of two
independent Euclidean fields, denoted in the code by `phi` and `phibar`. They are
not complex conjugates during the Euclidean solve. The physical complex field is
recovered only on the turning slice used as initial data for the real-time
Minkowski evolution.

The folder implements the following chain:

1. define the scalar potential and its derivatives;
2. solve reduced one-dimensional $O(d)$ bounce equations for $d=1,3,4$;
3. build two-dimensional seeds on the half-domain $(r,\tau)$;
4. solve the fixed-$Q$ Euclidean boundary-value problem with Newton iteration;
5. tune the residual twist `eta0` until the target charge is matched;
6. compute the fixed-charge exponent
   $F_{Q,\beta}=S_E[\phi_b,\bar\phi_b]-S_E[\phi_i,\bar\phi_i]+\eta_0 Q$;
7. perform charge, energy, and action diagnostics;
8. continue the turning slice to Minkowski time;
9. analyse long-time wall motion, charge rearrangement, and
   terminal-velocity diagnostics.

This is research code for the fixed-charge single-field model used in
*Globally Charged Vacuum Decay*. It is not meant to be a general finite-density
tunnelling package.

---

## Directory layout

```text
Bubble_finder/
├── potential_bubble.py
├── bounce_1d.py
├── ansatz_bubble.py
├── bounce2d.py
├── observables_1d.py
├── observables_2d.py
├── rate_exponent.py
├── branching.py
├── minkowski_evolution_post_tunneling.py
├── diagnostics_sanity.py
├── clean_analysis_helpers.py
├── notebooks/
│   ├── *.ipynb
│   └── figures/
└── tests/
    └── test_bubble_solver_2d.py
```

The modules are deliberately split according to the physics workflow. The
Euclidean fixed-charge solver is in `bounce2d.py`; everything else either
prepares inputs for it, computes observables from its output, or evolves the
turning slice after the tunnelling event.

---

## Core conventions

### Physical modulus versus solver amplitude

The paper writes the complex scalar as

$$
\Phi = {\rho \over \sqrt{2}} e^{i\theta} .
$$

The solver evolves the Euclidean fields

$$
\phi,\qquad \bar\phi,
$$

whose homogeneous amplitude is therefore

$$
|\phi_i| = {\rho_i \over \sqrt{2}} .
$$

Important naming convention: in `bounce2d.py` this solver amplitude is stored as
`rho0`, for historical reasons. Thus

```text
solver.rho0 = |phi_i| = rho_i/sqrt(2)
```

not the physical modulus $\rho_i$. Whenever the physical modulus is needed, the
code reconstructs

$$
\rho_{\rm phys} = \sqrt{2\,\phi\bar\phi} .
$$

This is the main convention to keep in mind when comparing the code to the
paper.

### Rotated Euclidean fields

The homogeneous charged state rotates in Minkowski time. After Wick rotation,
the Euclidean background contains exponential factors,

$$
\phi_i(\tau)=\phi_i e^{+\omega\tau},\qquad
\bar\phi_i(\tau)=\phi_i e^{-\omega\tau} .
$$

To remove this trivial exponential growth, the solver works with rotated fields,

$$
\phi_{\rm rot}=e^{-\omega\tau}\phi,\qquad
\bar\phi_{\rm rot}=e^{+\omega\tau}\bar\phi .
$$

The Newton unknowns are the shifted, regularized fields

$$
y = r(\phi_{\rm rot}-\phi_i),\qquad
\bar y = r(\bar\phi_{\rm rot}-\phi_i) .
$$

The homogeneous charged background is therefore exactly

```text
y = ybar = 0
```

on the numerical grid.

The residual twist `eta0` is the part of the Euclidean-time twist left after
removing the homogeneous contribution $\omega\beta$. It is the variable tuned by
the fixed-charge outer loop.

### Twist acts on total fields

The twist must be applied to the total rotated fields,

$$
y_{\rm tot}=y+r\,\phi_i,
\qquad
\bar y_{\rm tot}=\bar y+r\,\phi_i,
$$

not to the fluctuations alone. In code this appears in the half-box ghost rule
at $\tau=-\beta/2$ as

```python
y_ghost    = exp(-eta0) * (ybar + r*rho0) - r*rho0
ybar_ghost = exp(+eta0) * (y    + r*rho0) - r*rho0
```

This is not cosmetic. Twisting only `y` and `ybar` would impose the wrong
fixed-charge boundary condition and usually destroys Newton convergence.

### Half-domain in Euclidean time

The Euclidean solver stores only

$$
r \in (0,L_r),\qquad \tau \in (-\beta/2,0) .
$$

The other half is reconstructed by reflection and exchange,

$$
\phi(\tau,r)=\bar\phi(-\tau,r),\qquad
\bar\phi(\tau,r)=\phi(-\tau,r) .
$$

The slice $\tau=0$ is the turning slice. At this slice the data are matched to a
physical complex field and used as initial data for the Minkowski evolution.

---

## Module map

### `potential_bubble.py`

Defines the explicit scalar potential used in the numerical examples, together
with first and second derivatives. It also defines the reduced rotating-frame
potential

$$
\Omega(\rho;\omega)=V(\rho)-{1\over2}\omega^2\rho^2,
$$

and helpers such as `vacua_of_Omega`, `Omega_phi`, `dOmega_dphi`, and
`d2Omega_dphi2`. This module is the model input for both the 1D reduced bounces
and the 2D fixed-charge solver.

### `bounce_1d.py`

Solves the reduced $O(d)$ bounce equation in the rotating-frame potential for
$d=1,3,4$. These 1D solutions are used mainly as controlled seed profiles for
the full 2D Newton solve:

- $d=4$: seed connected to the zero-temperature Coleman bounce;
- $d=3$: static critical-bubble seed, useful for the thermal/static branch;
- $d=1$: Euclidean-time seed, useful for mapping non-bubble branches.

The solver uses shooting/overshoot-undershoot logic and `scipy.solve_ivp`.

### `ansatz_bubble.py`

Builds the two-dimensional initial guesses for `bounce2d.py`. It embeds 1D
profiles on the $(r,\tau)$ half-domain, constructs O(4)-like, O(3)-static,
O(1)-like, and homogeneous seeds, and converts them into the Newton variables
`y`, `ybar`.

This file also contains utilities for seed scoring and negative-mode based
experiments. These are useful for branch diagnostics but are not the primary
production path.

### `bounce2d.py`

Main Euclidean fixed-charge solver.

It defines:

- `Bubble2DSettings`: grid size, box size, $\beta$, reference frequency, twist,
  boundary-condition choices, Newton tolerances, and regularization choices;
- `Bubble2DSolution`: container for the converged fields, settings,
  observables, and Newton result;
- `Bubble2DSolver`: residual operator, sparse analytic Jacobian, packing and
  unpacking of real/imaginary components, ghost boundary conditions, Newton
  driver, eta scans, and fixed-charge matching.

The bulk residual is the rotated-field version of the Euclidean equations for
independent $\phi$ and $\bar\phi$. In complex-saddle mode the unknown vector is
split as

```text
Re(y), Im(y), Re(ybar), Im(ybar)
```

so the Newton system is a real sparse system. This is necessary because the
potential depends on $\mathrm{Re}(\phi\bar\phi)$ and the residual is not
holomorphic in the complex unknowns.

### `observables_1d.py`

Computes one-dimensional observables: charge, Minkowski energy,
grand-canonical functional $F_\omega=E-\omega Q$, density profiles, and
homogeneous-ball reference values. These functions are used to diagnose the 1D
reduced profiles before they are embedded as 2D seeds.

### `observables_2d.py`

Computes charge and energy diagnostics for the 2D Euclidean saddle. The most
important functions use the same $\tau=0$ ghost reconstruction as the solver, so
that derivatives on the turning slice are consistent with the imposed boundary
conditions.

Common diagnostics include:

- `compute_charge_tau0_ghost_2d`;
- `compute_energy_minkowski_tau0_ghost_2d`;
- `compute_HE_euclidean_tau0_ghost_2d`;
- `compute_observables_tau0_ghost`;
- `delta_E_M_tau0_ghost_2d`;
- charge-density and energy-density profiles.

By default some routines subtract the homogeneous background. Check the keyword
`subtract_background` before comparing absolute charges with target charges.

### `rate_exponent.py`

Computes the Euclidean action and the fixed-charge tunnelling exponent. Since
the numerical solution is stored only on the half-domain, the half-action is
doubled using the reflection symmetry. The physical exponent is then assembled
as

$$
F_{Q,\beta}=S_E^{\rm bounce}-S_E^{\rm hom}+\eta_0 Q .
$$

The `eta0*Q` term is not optional: it is the Legendre/projection term required
by the fixed-charge path integral.

### `branching.py`

Tools for exploring nearby Euclidean branches. In particular, this module
contains utilities to leave a sphaleron-like/static branch along a negative-mode
direction and to inspect slicewise energies. It is useful for branch mapping and
for diagnosing whether a Newton solution is the O(4)-connected tunnelling
solution or another nontrivial saddle.

### `minkowski_evolution_post_tunneling.py`

Spherically symmetric real-time evolution after tunnelling. The initial data are
extracted from the Euclidean turning slice using the same ghost reconstruction
as the Euclidean solver. The evolution is performed for the complex fields
`phi`, `phibar` with a velocity-Verlet integrator.

This module provides:

- extraction of $\phi$, $\bar\phi$, $\dot\phi$, $\dot{\bar\phi}$ from the
  turning slice;
- spherical finite-volume/finite-difference operators;
- Minkowski acceleration and velocity-Verlet update;
- optional sponge damping near the outer boundary;
- energy and charge diagnostics;
- polar diagnostics such as $\rho_{\rm phys}$ and the phase.

The modulus is at a turning point at $t=0$, but the phase is not. The phase
velocity carries the conserved charge and is essential for the subsequent
real-time dynamics.

### `diagnostics_sanity.py`

Independent checks of the Euclidean solver. The key tests are:

- residual on the homogeneous background;
- potential-convention checks;
- field-to-potential convention checks;
- finite-difference versus analytic Jacobian checks;
- boundary-condition and twist-source consistency checks.

Use this file when changing `bounce2d.py`; small sign or normalization mistakes
usually show up here before they show up in a full production run.

### `clean_analysis_helpers.py`

Notebook-facing helper functions: Matplotlib style, cache/load wrappers, 1D
scan utilities, unified seed-building calls, eta-scan wrappers, solution
summaries, and common plotting routines. The purpose is to keep the notebooks
readable; physics-critical operations remain in the modules above.

### `tests/`

Pytest checks for the 2D solver. They are intentionally small and fast. They do
not reproduce the full paper figures; they check that the core numerical wiring
has not been broken.

Run from the project root:

```bash
pytest Bubble_finder/tests -v
```

or from inside `Bubble_finder/`:

```bash
pytest tests -v
```

---

## Workflow in practice

A typical fixed-charge Euclidean run has this structure:

```python
from Bubble_finder.potential_bubble import V_phi, dV_dphi, d2V_dphi2
from Bubble_finder.bounce2d import Bubble2DSettings, Bubble2DSolver, make_potential_from_V
from Bubble_finder.ansatz_bubble import build_seed

U, dU, d2U = make_potential_from_V(
    V_phi, dV_dphi, d2V_dphi2,
    phi0=1.999,
    v1=1.0,
    v2=2.0,
)

settings = Bubble2DSettings(
    Nr=160,
    Ntau=320,
    Lr=15.0,
    beta=30.0,
    omega_ref=0.95,
    eta0=0.0,
    tau_bc="twisted",
    r_bc="neumann",
    rho0_bracket=(1.0, 1.3),
)

solver = Bubble2DSolver(settings, U, dU, d2U)
```

Then one usually:

1. computes or loads the 1D $O(d)$ profiles;
2. builds an O(4)-like seed;
3. solves the 2D Newton problem at a trial `eta0`;
4. scans/matches `eta0` to impose the target charge;
5. computes action, energy, and charge diagnostics;
6. extracts the turning slice;
7. evolves it with `minkowski_evolution_post_tunneling.py`.

The full version of this workflow is implemented in the notebooks.

---

## Notebooks and figures

The folder `notebooks/` contains the self-contained analysis notebooks used to
reproduce and check the results of *Globally Charged Vacuum Decay*. Run them in
the order described by the parent README.

Figures are written to

```text
Bubble_finder/notebooks/figures/
```

The code distinguishes between two types of notebook cells:

- expensive scan cells, which generate caches and figures;
- load/post-processing cells, which read cached results and regenerate plots.

For public notebooks, a full `Run All` should not start a long scan unless the
corresponding `RUN_*` flag is explicitly set.

---

## Real-time simulations

Real-time evolution is implemented in
`minkowski_evolution_post_tunneling.py`. It evolves the complex field on a
spherical grid starting from the fixed-charge Euclidean turning slice.

The main diagnostics are:

- total charge conservation;
- total energy conservation;
- bubble radius and wall velocity;
- charge-density redistribution;
- phase-gradient energy near the wall.

The real-time simulations are exploratory and more sensitive to box size,
time-step choice, sponge settings, and boundary reflections than the Euclidean
solver. Do not interpret late-time data unless energy and charge drifts have
been checked.

---

## Tests and sanity checks

Minimal tests:

```bash
pytest Bubble_finder/tests -v
```

Useful manual checks before trusting a production run:

1. `solver.sanity_background()` gives a small residual at `eta0=0`;
2. `solver.check_jacobian(...)` agrees with finite differences;
3. the charge computed at the turning slice matches the target charge;
4. the Euclidean/Minkowski energy on the turning slice agrees with the
   homogeneous reference within numerical accuracy;
5. the $Q\to0$ limit approaches the neutral O(4) bounce;
6. the large-$\beta$ result is stable under further increases of $\beta$;
7. real-time energy and charge drifts remain controlled before extracting
   velocities.

---

## Dependencies

The package uses standard scientific Python libraries:

```text
numpy
scipy
matplotlib
pytest
```

The Euclidean solver can reuse infrastructure from `Q_ball_finder` when it is
available. Internal fallbacks exist for the grid and Newton solver, but the full
project should be run from the repository root so that sibling packages are on
`PYTHONPATH`.

Typical invocation from the repository root:

```bash
python -m pytest Bubble_finder/tests -v
```

For notebooks, start Jupyter from the repository root so that imports such as

```python
from Bubble_finder.bounce2d import Bubble2DSolver
```

resolve without modifying `sys.path` inside every notebook.

---

## What is not implemented

This folder does not currently include:

- fluctuation determinants or prefactors around the fixed-charge saddle;
- multifield fixed-charge tunnelling;
- gauge constraints or local charge neutrality;
- a genuine fixed-$\mu$ open-system tunnelling solver;
- realistic dense-QCD equations of state;
- a production-level gravitational-wave spectrum calculator for the charged
  collisions.

Those are possible extensions. The present code is the fixed-$Q$ benchmark
pipeline for one global-$U(1)$ scalar field.

---

## Development notes

Keep the following points fixed unless you deliberately want to change the
numerical problem:

- apply the twist to `y + r*rho0`, not to `y` alone;
- keep `phi` and `phibar` independent during the Euclidean solve;
- use the same ghost reconstruction in observables as in the residual;
- distinguish solver amplitude `rho0` from physical modulus
  `rho_phys = sqrt(2 phi phibar)`;
- include the `eta0*Q` term when computing the fixed-charge exponent;
- check the homogeneous background residual after changing potential
  conventions;
- check energy and charge conservation before interpreting Minkowski outputs.

These are the places where sign and normalization errors most often enter.
