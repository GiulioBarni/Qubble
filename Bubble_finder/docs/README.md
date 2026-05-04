# Numerical implementation documentation

This directory is reserved for the standalone technical documentation of the
numerical implementation used in `Bubble_finder/`.

## Expected file

```text
Bubble_finder/docs/numerical_implementation.pdf
```

## Intended contents

The document corresponds to the technical details of the paper
*"Tunneling decay at fixed charge: nucleation theory"* (Barni & Espinosa),
rewritten as a self-contained numerical-implementation
note. It is intended to make the code in `Bubble_finder/` readable on its own,
without requiring the reader to keep the published paper open.

The PDF should cover:

1. domain and grid (half $\tau$-interval, half-step staggering, $r$-domain
   regularity);
2. rotated variables $\phi_{\rm rot} = e^{-\omega\tau}\phi$,
   $\bar\phi_{\rm rot} = e^{+\omega\tau}\bar\phi$, and the shifted-radial
   unknowns $y, \bar y$;
3. residual equations and the analytic sparse Jacobian, with the real/imag
   splitting used for complex saddles;
4. half-box boundary conditions: Neumann at $r=L_r$, regularity at $r=0$,
   reflection / swap at $\tau = 0$, twisted closure at $\tau = -\beta/2$
   acting on the **total** fields $y_{\rm tot} = y + r\rho_0$;
5. the Newton solver with backtracking line search;
6. the fixed-charge outer loop in the residual twist $\eta_0$;
7. ansatz construction (O(4)/O(3)/O(1)/homogeneous seeds and the
   sphaleron-aware augmented system);
8. observables and diagnostics: $\tau=0$ ghost reconstruction,
   slicewise charges and energies, sanity checks against the homogeneous
   background;
9. the post-tunnelling Minkowski evolution and its diagnostics
   (wall velocity, phase-gradient energy, polar decomposition).

## How to obtain or regenerate the PDF

Until the document is added, please refer to the corresponding appendices of
the paper. Once it is finalised, the PDF will be added directly to this
folder.
