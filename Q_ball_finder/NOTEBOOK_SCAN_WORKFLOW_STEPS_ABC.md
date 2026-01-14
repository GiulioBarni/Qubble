# Notebook Cell Snippets for Steps A, B, C Implementation

This document provides notebook cell snippets for the updated Q-ball bounce solver workflow implementing Steps A, B, and C to match the paper conventions (arXiv:1711.05279v2).

## Step A: Grid Conventions

Grid conventions are already correct in the package. The default `Lr=20.0` is set in `QBall2DSettings`. No notebook changes needed.

**Note**: If you want to verify Lr:
```python
from Q_ball_finder.bounce2d import QBall2DSettings
settings = QBall2DSettings()
print(f"Default Lr = {settings.Lr}")  # Should be 20.0
```

## Step B: Energy Computation (Eq. 2.4)

The new energy computation is automatically computed when solving. Access it via:

```python
# After solving, the solution now includes:
solution.energy_eq24_median      # Recommended: median energy from Eq.(2.4)
solution.energy_eq24_tau0        # Energy at tau≈0 from Eq.(2.4)
solution.energy_eq24_spread      # Energy spread (max-min)/mean
solution.energy_ratio_eq24       # Recommended: energy_eq24_median / E_Q

# Legacy fields (still available):
solution.energy_tau0             # Legacy: from compute_energy_slice
solution.energy_ratio            # Legacy: energy_tau0 / E_Q
```

## Step C: Beta Continuation with Proper Warm Start

### Helper Function: Compute Ntau for Beta

```python
from Q_ball_finder.beta_scan_continuation import compute_ntau_for_beta

# Reference values
beta_ref_scan = beta_ref  # Your reference beta
ntau_ref_scan = Ntau_ref  # Your reference Ntau

# For a given beta_i
beta_i = 60.0
ntau_i = compute_ntau_for_beta(beta_i, beta_ref_scan, ntau_ref_scan, min_ntau=100)
dtau_i = beta_i / (2.0 * ntau_i)
print(f"β = {beta_i:.4f}, Ntau = {ntau_i}, dtau = {dtau_i:.6f}")
```

### Helper Function: Warm Start from Solution

```python
from Q_ball_finder.notebook_utils import warm_start_from_solution
from Q_ball_finder.grid import build_grid

# Previous converged solution
solution_prev = result_Lr30_eta.solution  # or your previous solution

# New grid with updated beta
ntau_new = compute_ntau_for_beta(beta_new, beta_ref_scan, ntau_ref_scan)
grid_new = build_grid(nr_ref_scan, ntau_new, lr_ref_scan, beta_new)

# Prepare warm start
x0_initial = warm_start_from_solution(
    solution_prev,
    grid_new,
    omega_qball,  # Your omega parameter
    enforce_plateau=True,
)
```

### Complete Beta Scan Loop

```python
from Q_ball_finder.bounce2d import QBall2DSettings, scan_eta_to_match_charge
from Q_ball_finder.grid import build_grid
from Q_ball_finder.beta_scan_continuation import compute_ntau_for_beta
from Q_ball_finder.notebook_utils import warm_start_from_solution
from Q_ball_finder.bounce2d import compute_charge_profile_eq39

# --- Beta scan with dtau-fixed grids and proper warm starts ---

# Reference values for dtau calculation
beta_ref_scan = beta_ref  # Use your reference beta
ntau_ref_scan = Ntau_ref  # Use your reference Ntau
nr_ref_scan = Nr_ref      # Use your reference Nr
lr_ref_scan = 20.0        # Paper default (or use Lr_ref if different)

# Beta scan parameters - use small steps for continuation
beta_step = 0.5  # Small step for continuation (0.2-0.5 recommended, paper uses small steps)
num_steps = 10   # Number of steps
beta_start = beta_ref
beta_values_scan = beta_start + beta_step * np.arange(num_steps + 1)

# Storage
eta_values_scan = []
energy_ratios_scan = []
energy_eq24_medians = []
energy_spreads = []
solutions_scan = []
ntau_values = []
dtau_values = []
charge_values = []
charge_spreads = []

# Start from the Lr=30 eta scan result (or your starting solution)
solution_current = result_Lr30_eta.solution  # Your starting solution
eta_current = result_Lr30_eta.eta

for idx, beta_i in enumerate(beta_values_scan):
    print(f"\n{'='*60}")
    print(f"Beta scan step {idx+1}/{len(beta_values_scan)}: β = {beta_i:.4f}")
    print(f"{'='*60}")
    
    # Compute Ntau for this beta to keep dtau approximately constant
    ntau_i = compute_ntau_for_beta(beta_i, beta_ref_scan, ntau_ref_scan, min_ntau=100)
    dtau_i = beta_i / (2.0 * ntau_i)
    
    print(f"Ntau = {ntau_i}, dtau = {dtau_i:.6f}")
    
    # Build grid for this beta
    grid_i = build_grid(nr_ref_scan, ntau_i, lr_ref_scan, beta_i)
    
    # Prepare warm start: ALWAYS resample to handle grid differences
    # This handles both Lr changes (e.g., Lr=30 to Lr=20) and beta/Ntau changes
    x0_current = warm_start_from_solution(
        solution_current,
        grid_i,
        omega_qball,  # Your omega parameter
        enforce_plateau=True,
    )
    
    # Settings for this beta
    snapshot_dir_i = PROJECT_ROOT / "Q_ball_finder" / "notebooks" / f"newton_beta_cont_{idx:02d}"
    settings_i = QBall2DSettings(
        beta=beta_i,
        ansatz_amplitude=0.8,  # Not used when x0_initial provided
        energy_reference=E_Q,
        omega_tilde=omega_cloud,
        Nr=nr_ref_scan,
        Ntau=ntau_i,
        Lr=lr_ref_scan,
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
    
    # Compute charge profile diagnostics using Eq.(3.9)
    Q_tau, Q_median, Q_mean, Q_std, charge_spread = compute_charge_profile_eq39(
        result_i.solution.y,
        result_i.solution.ybar,
        result_i.solution.grid,
        omega_qball,
        result_i.solution.settings.eta0,
    )
    
    # Store results
    eta_values_scan.append(result_i.eta)
    solutions_scan.append(result_i)
    ntau_values.append(ntau_i)
    dtau_values.append(dtau_i)
    charge_values.append(Q_median)
    charge_spreads.append(charge_spread)
    
    # Store energy results (Eq. 2.4 - recommended)
    if result_i.solution.energy_eq24_median is not None:
        energy_eq24_medians.append(result_i.solution.energy_eq24_median)
        energy_ratios_scan.append(result_i.solution.energy_ratio_eq24)
    else:
        energy_eq24_medians.append(None)
        energy_ratios_scan.append(None)
    
    if result_i.solution.energy_eq24_spread is not None:
        energy_spreads.append(result_i.solution.energy_eq24_spread)
    else:
        energy_spreads.append(None)
    
    # Print diagnostics
    print(f"\n--- Diagnostics for β = {beta_i:.4f} ---")
    print(f"η* = {result_i.eta:.6f}")
    print(f"Q (Eq.3.9) = {Q_median:.6f} (target: {g2Q_target:.6f}, ratio: {Q_median/g2Q_target:.6f})")
    print(f"Charge spread = {charge_spread:.6e}")
    if result_i.solution.energy_eq24_median is not None:
        print(f"Energy (Eq.2.4, median) = {result_i.solution.energy_eq24_median:.6f}")
        if result_i.solution.energy_ratio_eq24 is not None:
            print(f"Energy ratio (Eq.2.4) = {result_i.solution.energy_ratio_eq24:.6f}")
        if result_i.solution.energy_eq24_spread is not None:
            print(f"Energy spread = {result_i.solution.energy_eq24_spread:.6e}")
    
    # Update for next iteration
    solution_current = result_i.solution
    eta_current = result_i.eta

# Convert to arrays
eta_values_scan = np.array(eta_values_scan)
energy_ratios_scan = np.array(energy_ratios_scan)
energy_eq24_medians = np.array(energy_eq24_medians)
energy_spreads = np.array(energy_spreads)
ntau_values = np.array(ntau_values)
dtau_values = np.array(dtau_values)
charge_values = np.array(charge_values)
charge_spreads = np.array(charge_spreads)

print(f"\n{'='*60}")
print("Beta scan completed!")
print(f"{'='*60}")
```

### Plotting Results

```python
import matplotlib.pyplot as plt

# Plot energy ratio vs beta
fig, ax = plt.subplots(figsize=(8, 6))
mask = ~np.isnan(energy_ratios_scan)
ax.plot(beta_values_scan[mask], energy_ratios_scan[mask], marker="o", lw=2, markersize=6, label="Energy ratio (Eq.2.4)")
ax.axhline(y=1.0, color="r", linestyle="--", alpha=0.5, label="Target (E/E_Q = 1)")
ax.set_xlabel(r"$\beta$")
ax.set_ylabel(r"$E/E_Q$ (Eq. 2.4, median)")
ax.set_title("Energy Ratio vs Beta")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Plot energy spread vs beta
fig, ax = plt.subplots(figsize=(8, 6))
mask = ~np.isnan(energy_spreads)
ax.plot(beta_values_scan[mask], energy_spreads[mask], marker="o", lw=2, markersize=6, label="Energy spread")
ax.set_xlabel(r"$\beta$")
ax.set_ylabel("Energy spread (max-min)/mean")
ax.set_title("Energy Spread vs Beta")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Plot charge vs beta (should be constant)
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(beta_values_scan, charge_values, marker="o", lw=2, markersize=6, label="Charge (Eq.3.9)")
ax.axhline(y=g2Q_target, color="r", linestyle="--", alpha=0.5, label=f"Target (Q = {g2Q_target:.2f})")
ax.set_xlabel(r"$\beta$")
ax.set_ylabel(r"$Q$")
ax.set_title("Charge vs Beta")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

## Key Changes Summary

1. **Grid Conventions (Step A)**: Already correct in package, no changes needed.
2. **Energy Computation (Step B)**: 
   - Use `solution.energy_eq24_median` and `solution.energy_ratio_eq24` (recommended)
   - Legacy fields still available for backward compatibility
3. **Beta Continuation (Step C)**:
   - Always use `warm_start_from_solution()` to handle grid changes properly
   - Uses tau clamping to plateau to prevent drift
   - Resamples only y/ybar and reconstructs phi/phibar
   - Enforces plateau in extended tau regions
   - Keeps dtau approximately constant by adjusting Ntau



