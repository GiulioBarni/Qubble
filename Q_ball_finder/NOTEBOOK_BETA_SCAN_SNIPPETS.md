# Notebook Cell Snippets for Beta Scan with Proper Continuation

## Setup Cell (run once)

```python
# Import helper functions for beta scan continuation
from Q_ball_finder.beta_scan_continuation import (
    compute_ntau_for_beta,
    enforce_plateau_after_resample,
    prepare_warm_start_for_beta_continuation,
)
from Q_ball_finder.bounce2d import compute_charge_profile_eq39
from Q_ball_finder.ansatz import AnsatzResult
from Q_ball_finder.grid import build_grid, pack_fields
from Q_ball_finder.notebook_utils import resample_ansatz
```

## Beta Scan Cell (with dtau-fixed continuation)

```python
# --- Beta scan with dtau-fixed grids and proper warm starts ---

# Reference values for dtau calculation
beta_ref = 55.0 - 0.2  # Your reference beta
ntau_ref = 81 * 2  # Your reference Ntau
nr_ref = 48 * 2  # Your Nr
lr_ref = 15.0  # Your Lr

# Beta scan parameters
beta_step = 0.5  # Small step for continuation (0.2-0.5 recommended)
num_steps = 10  # Number of steps
beta_start = beta_ref
beta_values_scan = beta_start + beta_step * np.arange(num_steps + 1)

# Storage
eta_values_scan = []
energy_ratios_scan = []
solutions_scan = []
ntau_values = []
dtau_values = []
charge_values = []
charge_spreads = []

# Start from your initial solution (e.g., result_Lr30_eta or result_ref)
solution_current = result_Lr30_eta.solution  # Or your starting solution
eta_current = result_Lr30_eta.eta

for idx, beta_i in enumerate(beta_values_scan):
    print(f"\n{'='*60}")
    print(f"Beta scan step {idx+1}/{len(beta_values_scan)}: β = {beta_i:.4f}")
    print(f"{'='*60}")
    
    # Compute Ntau for this beta to keep dtau approximately constant
    ntau_i = compute_ntau_for_beta(beta_i, beta_ref, ntau_ref, min_ntau=100)
    dtau_i = beta_i / (2.0 * ntau_i)
    
    print(f"Ntau = {ntau_i}, dtau = {dtau_i:.6f}")
    
    # Build grid for this beta
    grid_i = build_grid(nr_ref, ntau_i, lr_ref, beta_i)
    
    # Prepare warm start (only if not first step)
    if idx == 0:
        # First step: use existing solution as-is
        x0_current = pack_fields(solution_current.y, solution_current.ybar)
    else:
        # Continuation: resample previous solution onto new grid
        ans_old = AnsatzResult(
            phi=solution_current.phi,
            phibar=solution_current.phibar,
            y=solution_current.y,
            ybar=solution_current.ybar,
        )
        
        # Resample onto new grid
        ans_new = resample_ansatz(ans_old, solution_current.grid, grid_i)
        
        # CRITICAL: Enforce plateau in new tau region (if beta increased)
        if beta_i > beta_values_scan[idx-1]:
            ans_new = enforce_plateau_after_resample(ans_new, ans_old)
        
        x0_current = pack_fields(ans_new.y, ans_new.ybar)
    
    # Settings for this beta
    snapshot_dir_i = PROJECT_ROOT / "Q_ball_finder" / "notebooks" / f"newton_beta_cont_{idx:02d}"
    settings_i = QBall2DSettings(
        beta=beta_i,
        ansatz_amplitude=0.8,
        energy_reference=E_Q,
        omega_tilde=omega_cloud,
        Nr=nr_ref,
        Ntau=ntau_i,
        Lr=lr_ref,
        newton_verbose=True,
        store_snapshots=True,
        snapshot_dir=str(snapshot_dir_i),
        snapshot_prefix=f"beta_cont_{idx:02d}",
        snapshot_overwrite=True,
    )
    
    # Use eta* from previous step as starting point
    eta_start_i = eta_current
    
    # Run eta scan
    result_i = scan_eta_to_match_charge(
        params,
        omega_qball,
        qcloud_profile,
        unstable_mode,
        settings=settings_i,
        eta_start=eta_start_i,
        target_charge=g2Q_target,
        d_eta=0.1,
        max_scan_steps=30,
        tol=1e-4,
        verbose=True,
        x0_initial=x0_current,
    )
    
    # Compute charge profile diagnostics
    Q_tau, Q_median, Q_mean, Q_std, charge_spread = compute_charge_profile_eq39(
        result_i.solution.y,
        result_i.solution.ybar,
        result_i.solution.grid,
        omega_qball,
        result_i.solution.settings.eta0,
    )
    
    # Store results
    eta_values_scan.append(result_i.eta)
    energy_ratios_scan.append(result_i.solution.energy_ratio)
    solutions_scan.append(result_i)
    ntau_values.append(ntau_i)
    dtau_values.append(dtau_i)
    charge_values.append(Q_median)
    charge_spreads.append(charge_spread)
    
    # Print diagnostics
    print(f"\n--- Diagnostics for β = {beta_i:.4f} ---")
    print(f"η* = {result_i.eta:.6f}")
    print(f"Q (Eq.3.9) = {Q_median:.6f} (target: {g2Q_target:.6f}, ratio: {Q_median/g2Q_target:.6f})")
    print(f"Charge spread = {charge_spread:.6e}")
    print(f"Ntau = {ntau_i}, dtau = {dtau_i:.6f}")
    if result_i.solution.energy_ratio is not None:
        print(f"Energy ratio = {result_i.solution.energy_ratio:.6f}")
    
    # Update for next iteration
    solution_current = result_i.solution
    eta_current = result_i.eta
    x0_current = pack_fields(solution_current.y, solution_current.ybar)

# Convert to arrays
eta_values_scan = np.array(eta_values_scan)
energy_ratios_scan = np.array(energy_ratios_scan)
ntau_values = np.array(ntau_values)
dtau_values = np.array(dtau_values)
charge_values = np.array(charge_values)
charge_spreads = np.array(charge_spreads)

print(f"\n{'='*60}")
print("Beta scan completed!")
print(f"{'='*60}")
```

## Diagnostic Plot Cell

```python
# --- Plot diagnostics from beta scan ---

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Energy ratio vs beta
ax = axes[0, 0]
mask = ~np.isnan(energy_ratios_scan)
ax.plot(beta_values_scan[mask], energy_ratios_scan[mask], marker="o", lw=2, markersize=6)
ax.set_xlabel(r"$\beta$")
ax.set_ylabel(r"$E(\tau\approx 0) / E_Q$")
ax.set_title("Energy ratio vs $\\beta$")
ax.grid(True, ls="--", alpha=0.6)

# dtau vs beta (should be approximately constant)
ax = axes[0, 1]
ax.plot(beta_values_scan, dtau_values, marker="o", lw=2, markersize=6)
ax.set_xlabel(r"$\beta$")
ax.set_ylabel(r"$d\tau$")
ax.set_title("$d\\tau$ vs $\\beta$ (should be ~constant)")
ax.grid(True, ls="--", alpha=0.6)

# Ntau vs beta
ax = axes[1, 0]
ax.plot(beta_values_scan, ntau_values, marker="o", lw=2, markersize=6)
ax.set_xlabel(r"$\beta$")
ax.set_ylabel(r"$N_\tau$")
ax.set_title("$N_\\tau$ vs $\\beta$")
ax.grid(True, ls="--", alpha=0.6)

# Charge spread vs beta
ax = axes[1, 1]
ax.plot(beta_values_scan, charge_spreads, marker="o", lw=2, markersize=6)
ax.set_xlabel(r"$\beta$")
ax.set_ylabel("Charge spread")
ax.set_title("Charge spread vs $\\beta$")
ax.grid(True, ls="--", alpha=0.6)

plt.tight_layout()
plt.show()
```

## Alternative: Using the Helper Function

If you prefer a more compact version using the helper function:

```python
# --- Beta scan using prepare_warm_start_for_beta_continuation helper ---

beta_step = 0.5
num_steps = 10
beta_start = beta_ref
beta_values_scan = beta_start + beta_step * np.arange(num_steps + 1)

eta_values_scan = []
energy_ratios_scan = []
solutions_scan = []

solution_current = result_Lr30_eta.solution
eta_current = result_Lr30_eta.eta

for idx, beta_i in enumerate(beta_values_scan):
    print(f"\nBeta step {idx+1}/{len(beta_values_scan)}: β = {beta_i:.4f}")
    
    if idx == 0:
        # First step: use existing solution
        grid_i = solution_current.grid
        x0_current = pack_fields(solution_current.y, solution_current.ybar)
        ntau_i = grid_i.Ntau
    else:
        # Continuation: prepare warm start
        x0_current, grid_i, ntau_i = prepare_warm_start_for_beta_continuation(
            solution_current,
            beta_i,
            beta_ref,
            ntau_ref,
            nr_ref,
            lr_ref,
            min_ntau=100,
        )
        print(f"Ntau = {ntau_i}, dtau = {grid_i.dtau:.6f}")
    
    # Settings and scan (same as above)
    # ... (rest of scan code)
```

## Important Notes

1. **CRITICAL**: The plateau enforcement is mandatory when beta increases. Without it, the new tau region will be filled with zeros, creating an artificial frontier.

2. **Do NOT rebuild negative-mode ansatz** for continuation steps. Always use the previous converged solution resampled to the new grid.

3. **Use small beta steps** (0.2-0.5) for better continuation. Large steps can cause convergence issues.

4. **Monitor charge spread**: If charge_spread becomes large (> 0.01), it may indicate issues with the solution or grid resolution.

5. **dtau should remain approximately constant** across the scan. If it drifts significantly, check the Ntau calculation.




