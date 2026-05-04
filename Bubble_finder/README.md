# Bubble finder

Main numerical implementation accompanying the paper *"Tunneling decay at
fixed charge: nucleation theory"* (Barni & Espinosa).

This subpackage contains the core fixed-$Q$ bubble-nucleation pipeline used to
generate the paper figures: the explicit scalar model, neutral Coleman
references, reduced 1D bounces, the two-dimensional fixed-$Q$ Euclidean
solver, the residual-twist outer loop, observables and diagnostics, the decay
exponent, and the post-tunnelling Minkowski continuation.

For a high-level overview of the project and installation instructions, see
the [root README](../README.md).

---

## Purpose

`Bubble_finder/` covers, in order of the workflow:

- the explicit scalar potential $V(\phi)$ used in the paper;
- neutral Coleman O(4) reference bounces obtained from a reduced
  $\Omega(\phi) = V(\phi) - \tfrac12\omega^2\phi^2$ formulation;
- reduced 1D O(d) bounce equations for $d \in \{1,3,4\}$;
- the **two-dimensional fixed-$Q$ Euclidean saddle** in $(r, \tau)$,
  formulated with independent fields $\phi$ and $\bar\phi$;
- charge projection and the resulting **twisted Euclidean-time boundary
  conditions**;
- rotated variables that remove the homogeneous Euclidean exponential and
  stabilise the Newton iteration;
- a Newton solver with analytic sparse Jacobian, backtracking line search,
  and complex-saddle real/imaginary splitting;
- the **fixed-charge outer loop** that tunes the residual twist $\eta_0$ to
  match a target charge $Q$;
- branch construction from O(4), O(3), and O(1) seeds, including a
  sphaleron-aware augmented system;
- static-energy decomposition and energy/charge diagnostics;
- the fixed-$Q$ decay exponent
  $F_{Q,\beta}^{\rm bounce} = S_E[\phi_b,\bar\phi_b] - S_E[\phi_i,\bar\phi_i] + \eta_0\,Q$;
- post-tunnelling **Minkowski continuation** from the turning slice
  $\tau=0$, including wall velocity and phase-gradient diagnostics;
- the publication figure pipeline.

---

## Physics conventions

The notation in this folder mirrors the paper. The most important
conventions are summarised here.

### Field variables

The model is a complex scalar with global U(1) symmetry. Three different
variables appear in the code:

| Variable | Meaning | Where it is used |
|---|---|---|
| `phi` (real) | scalar profile in the **1D reduced** problem | `bounce_1d.py`, `observables_1d.py` |
| `phi`, `phibar` (complex) | independent Euclidean fields treated as unrelated unknowns; physical reality is recovered when the saddle is real | `bounce2d.py`, `observables_2d.py` |
| `rho`, `rho_phys` | physical modulus $\rho = \sqrt{2\,\mathrm{Re}(\phi\bar\phi)}$ | observables, Minkowski evolution, plots |

In the 2D solver, the Newton unknowns are the **rotated variables**

$$
\phi_{\rm rot} = e^{-\omega_{\rm ref}\tau}\phi,\qquad
\bar\phi_{\rm rot} = e^{+\omega_{\rm ref}\tau}\bar\phi,
$$

shifted by the homogeneous background and multiplied by $r$:

$$
y = r(\phi_{\rm rot} - \rho_0),\qquad
\bar y = r(\bar\phi_{\rm rot} - \rho_0).
$$

This choice stabilises the iteration by making $y, \bar y \to 0$ on the
homogeneous background. The twist is applied to the **total** fields
$y_{\rm tot} = y + r\rho_0$, not to the fluctuations alone — this is the
single most important detail of the implementation.

### Background and frequency

| Symbol | Code name | Meaning |
|---|---|---|
| $\omega$ | `omega`, `omega_ref` | reference rotating-frame frequency selecting the metastable charged state |
| $\rho_0$ | `rho0` | homogeneous-background modulus, root of $V'(\rho) = 2\omega^2 \rho$ |
| $Q$ | `Q`, `target_charge` | total conserved global charge |
| $\eta_0$ | `eta0` | residual twist in Euclidean time, conjugate to $Q$ |
| $\beta$ | `beta` | Euclidean-time extent (full interval $[-\beta/2, \beta/2]$, of which only the half $(-\beta/2, 0]$ is solved) |

### Grid and boundary conditions

- $r$-grid: half-step grid, `r = (k + 1/2) dr`.
- $\tau$-grid: half-step grid on the **half-interval** $(-\beta/2, 0)$,
  `tau = -(k + 1/2) dtau`. The other half is recovered by reflection at
  $\tau = 0$ (the **turning slice**).
- $r$-boundary: Neumann at $r=0$ (regularity) and at $r=L_r$.
- $\tau$-boundary: **twisted closure** at $\tau = -\beta/2$ implemented via
  ghost rules acting on the total fields, with reflection / swap at
  $\tau = 0$.
- The default production setting is `tau_bc = "twisted"`,
  `r_bc = "neumann"`. Other BC combinations are available only with
  `settings.allow_debug_bcs=True`.

### Charge convention

Throughout this folder the Euclidean charge density is

$$
q_E(\tau,r) = \mathrm{Re}\!\left(\bar\phi\,\partial_\tau\phi - \phi\,\partial_\tau\bar\phi\right),
$$

so that

$$
Q(\tau) = 4\pi \int_0^{r_{\max}} dr\, r^2\, q_E(\tau, r).
$$

For the 1D O(3) bounce in the rotating-phase ansatz $\phi(r)e^{i\omega t}$
this reduces to $Q = 4\pi\omega \int dr\, r^2\, \phi(r)^2$, consistent with
the 2D ghost reconstruction at $\tau=0$.

### Continuation to real time

The Minkowski continuation uses $t = -i\tau$, i.e.

$$
\partial_t\phi\big|_{t=0} = +i\,\partial_\tau\phi\big|_{\tau=0},\qquad
\partial_t\bar\phi\big|_{t=0} = +i\,\partial_\tau\bar\phi\big|_{\tau=0},
$$

evolved with a velocity-Verlet integrator. A polar decomposition
$\phi = (\rho_{\rm phys}/\sqrt2)\,e^{+\beta}$,
$\bar\phi = (\rho_{\rm phys}/\sqrt2)\,e^{-\beta}$ is used for diagnostics
only.

---

## Detailed numerical implementation

A self-contained technical explanation of the numerical implementation is
provided in:

```text
Bubble_finder/docs/numerical_implementation.pdf
```

This document corresponds to the technical appendices of the paper
(Appendix B onward), rewritten as standalone documentation. It explains the
domain and grid, rotated variables, residual equations, half-box boundary
conditions, Newton solver, fixed-charge outer loop, ansatz construction,
diagnostics, observables, and the post-tunnelling Minkowski evolution.

If the file is not present in the cloned repository, see
[`docs/README.md`](docs/README.md) for an explanation.

---

## File overview

| File | Role |
|---|---|
| `__init__.py` | namespace marker (kept intentionally minimal). |
| `potential_bubble.py` | scalar potential $V(\phi) = -\tfrac12 + (\phi-1)^2[2\phi-5 + (2-\phi)^2\log(\dots)]$ used in the paper, plus first/second derivatives, the rotating-frame potential $\Omega(\phi) = V(\phi) - \tfrac12\omega^2\phi^2$, and the helpers `vacua_of_Omega`, `dOmega_dphi`, `Omega_phi`. False vacuum at $\phi = v_1 = 1$, true vacuum at $\phi = v_2 = 2$, with reference $\phi_0 = 1.999$. |
| `bounce_1d.py` | reduced O(d)-symmetric 1D bounce solver in the rotating frame. Uses overshoot/undershoot bisection on $\phi(0)$ via `scipy.solve_ivp` with events. Returns radial profile, derivatives, charge, energy, and grand-canonical functional. |
| `observables_1d.py` | 1D charge $Q$, Minkowski energy $E_M$, grand-canonical functional $F_\omega = E_M - \omega Q$, density profiles, and finite-volume homogeneous-ball references. |
| `bounce2d.py` | the **main 2D Euclidean solver**. Defines `Bubble2DSettings`, `Bubble2DSolver`, the residual operator, the analytic sparse Jacobian (real/imag split for complex saddles), the Newton driver with backtracking, the homogeneous-background sanity check, the $\eta_0$ scan and the eta-matching outer loop. Reuses `Q_ball_finder.grid` and `Q_ball_finder.nr_solver` when available, with internal fallbacks otherwise. |
| `ansatz_bubble.py` | construction of seed configurations: O(4)-inspired (Coleman-rotation), O(3)-static, O(1)-$\tau$-only, and homogeneous seeds. Includes 1D-to-2D embedding helpers and parameter sweeps. |
| `branching.py` | branch finding from a sphaleron-like 2D root: augmented system pinning a step along the negative-mode direction to leave the basin of the static saddle. Provides slicewise energies and recentring utilities. |
| `observables_2d.py` | 2D charge and energy with the **$\tau=0$ ghost reconstruction** (twisted-BC-aware derivatives at the turning slice), free-energy decomposition, surface-tension diagnostics. |
| `rate_exponent.py` | computation of the Euclidean action $S_E$ and of the fixed-$Q$ suppression exponent $F_{Q,\beta}^{\rm bounce} = S_E[\phi_b] - S_E[\phi_i] + \eta_0 Q$. The half-interval action is doubled by reflection symmetry. |
| `minkowski_evolution_post_tunneling.py` | post-tunnelling real-time evolution of the full complex fields $\phi$, $\bar\phi$ from the turning slice $\tau=0$. Provides the velocity-Verlet integrator, the regular spherical Laplacian, an outer sponge to damp outgoing radiation, optional symmetrisation $\bar\phi=\phi^*$, and Minkowski-energy/wall diagnostics. |
| `diagnostics_sanity.py` | independent sanity checks: residual on the homogeneous background, finite-difference Jacobian agreement, complex-saddle real/imag wiring, twist-source predictions at $\eta_0 \neq 0$. |
| `clean_analysis_helpers.py` | reusable plotting and bookkeeping utilities for the main notebook (style setup, 1D bounce caching, $\omega$ scans, unified seed interface, $\eta_0$-scan wrapper, slicewise static-energy diagnostics). |
| `notebooks/Bubble_Tunneling_2D.ipynb` | **the main paper notebook**. Sections 1–17 cover the full pipeline. |
| `notebooks/figures/` | output directory for paper-ready PDFs and development PNGs. |
| `docs/` | placeholder directory for the technical implementation PDF (see `docs/README.md`). |
| `tests/test_bubble_solver_2d.py` | pytest suite for the 2D solver: background exactness, Jacobian FD check, a few Newton steps from a clean ansatz. |

---

## Main workflow

The recommended end-to-end workflow, mirroring the section structure of the
main notebook, is:

1. **Define model parameters** — $V(\phi)$ from `potential_bubble.py`,
   reference frequency $\omega$, and the homogeneous background $\rho_0$
   from `solve_rho0_for_omega`.
2. **Compute / load the neutral Coleman reference** — 1D O(4) bounce at
   $\omega = 0$ via `bounce_1d.solve_bounce`.
3. **Compute reduced 1D bounce profiles** at the chosen $\omega$ for
   $d = 1, 3, 4$ for use as 2D seeds.
4. **Build O(4), O(3), and O(1) seeds** with `ansatz_bubble.py`. Each seed
   provides $(y_0, \bar y_0)$ on the 2D solver grid.
5. **Solve the 2D fixed-$Q$ Euclidean BVP** with `Bubble2DSolver.solve` from
   each seed. This uses Newton with analytic sparse Jacobian, line search,
   and the twisted closure at $\tau = -\beta/2$.
6. **Tune the residual twist $\eta_0$** with the eta-scan / eta-matching
   utilities to match the target charge $Q$.
7. **Compute the fixed-$Q$ exponent** with `rate_exponent.py`, using the
   homogeneous reference $S_E[\phi_i,\bar\phi_i] = \beta V_{\rm ball}(V(s_0) - \omega^2 s_0)$
   with $s_0 = \rho_0^2/2$.
8. **Check energy and charge conservation** with the `observables_2d.py`
   diagnostics, comparing the $\tau=0$ ghost reconstruction with the bulk
   integrals.
9. **Compare the charged static barrier** with the neutral Coleman barrier
   (the static-energy comparison in the paper).
10. **Continue the turning slice to real time** with
    `minkowski_evolution_post_tunneling.py`.
11. **Extract wall velocity and wall-energy diagnostics** from the Minkowski
    evolution.
12. **Generate the paper figures** with the "Plots for paper" subsections of
    the notebook.

The "Plots for paper" subsections are intentionally pure post-processing:
they reorganise quantities computed in the analysis cells above into
publication-ready figures and save them under `notebooks/figures/`. They do
not introduce new physics input.

---

## Figures

All figures generated by the main notebook are saved under
`Bubble_finder/notebooks/figures/`. PDFs are intended for the paper; PNGs
are kept for development traceability.

A non-exhaustive map between the paper figures and the files in this folder:

| Paper topic | File(s) |
|---|---|
| potential and rotating-frame vacua | `potential_and_omega_vacua.pdf`, `scalar_potential_and_omega.pdf` |
| 1D O(d) profiles | `bounce_profiles_fixed_and_zero_omega.pdf`, `one_dimensional_density_profiles.pdf`, `one_dimensional_omega_scan_observables.pdf`, `one_dimensional_profile_evolution.pdf` |
| 2D O(4) solution and seeds | `seed_O4_vs_1d_profiles.pdf`, `seed_O4_overview.pdf`, `o4_red_solution_map.pdf`, `o4_red_vs_1d_profiles.pdf`, `o4_fancy_full_symmetry_map.pdf` |
| static-energy comparison | `o4_energy_static.pdf`, `o4_energy_static_Qneq0_vs_Coleman_Q0.pdf`, `static_energy_*.pdf` |
| energy decomposition vs $\rho_{\rm phys}$, $\beta$ | `o4_energy_decomposition_rho_phys_beta_except_rho_tau.pdf` |
| O(3) and O(1) parallel analyses | `comparison_O3_static_seed.pdf`, `seed_O3_static_seed.pdf`, `solution_O3_static_seed.pdf`, `comparison_O1_tau_seed.pdf`, `seed_O1_tau_seed.pdf` |
| $\beta$-continuation | `beta_continuation_decay_exponent.pdf`, `beta_continuation_energy_ratios.pdf`, `beta_continuation_energy_and_decay_panel.pdf` |
| $\omega$-scan / decay rate | `decay_rate_vs_beta.pdf`, `o4_omega_scan_decay_exponent_vs_Q_only.pdf`, `omega_scan_velocity_estimators.pdf`, `omega_scan_velocity_terminal_fit_band.pdf` |
| interface and charge flow | `o4_interface_and_charge_flow.pdf`, `o4_interface_and_charge_flow_solution_vs_seed.pdf`, `asymptotic_velocity_and_surface_tension_vs_radius.pdf` |
| Minkowski post-tunnelling evolution | `minkowski_combined_evolution.pdf`, `minkowski_combined_evolution_omega_eff.pdf`, `minkowski_all_energy_contributions.pdf`, `minkowski_arg_phi_over_t.pdf`, `minkowski_arg_unwrap_over_t.pdf`, `minkowski_im_dphi_over_phi.pdf`, `minkowski_phase_direction_segments.pdf`, `minkowski_phase_energy_physical.pdf`, `stable_terminal_force_model.pdf` |

---

## Cached solutions

The `Bubble_finder/` pipeline currently does **not** ship cached `.npz`
solutions: the analysis is regenerated end-to-end from the main notebook.
Some cells are computationally expensive (continuation scans in $\beta$ and
$\omega$), so re-running the full notebook on a workstation can take
non-trivial time.

If cached 2D solutions are added in the future, the convention used in the
sister folder `Q_ball_finder/data/` should be followed: `.npz` files keyed
by the parameter being varied, gitignored (`*.npz`) and archived externally.

---

## Tests

The local test suite is run with:

```bash
pytest Bubble_finder/tests
```

It contains:

- `test_sanity_background` — residual on the homogeneous background must be
  $< 10^{-6}$ at $\eta_0 = 0$, exercising the bulk equations and the
  twisted-BC wiring;
- `test_check_jacobian` — analytic Jacobian against finite-difference
  directional derivatives, with $\max\,\text{rel}<2\%$;
- `test_demo_newton_few_steps` — a short Newton run from the standard
  anchored ansatz (no convergence required).

These tests are fast (seconds) and do not require any data files.

---

## Known limitations

- This is **research code**, tailored to the specific scalar model and
  numerical study in the paper. It is not a general-purpose finite-density
  tunnelling package.
- The implementation targets a single scalar field with a global U(1).
  Multifield dynamics, gauge fields, neutrality conditions, and realistic
  QCD-like applications are out of scope of the current code.
- One-loop fluctuation determinants / decay-rate prefactors are **not**
  computed: only the semiclassical exponent and its derivatives are
  produced.
- Fixed-$\mu$ open-system tunnelling is discussed in the paper but is not
  implemented in this folder. The implementation is genuinely fixed-$Q$.
- Bitwise reproducibility across machines, BLAS variants, and library
  versions is not guaranteed. Numerical agreement to several significant
  digits is the relevant bar; the test suite checks this on small
  configurations.
