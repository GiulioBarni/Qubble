"""
Construction of the 2D ansatz used to initialise the Newton solver.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .grid import RadialTimeGrid, phi_from_y, phi_from_ybar
from .profiles import QBallProfile, UnstableMode


@dataclass
class AnsatzResult:
    phi: np.ndarray
    phibar: np.ndarray
    y: np.ndarray
    ybar: np.ndarray


def build_negative_mode_ansatz(
    profile: QBallProfile,
    mode: UnstableMode,
    grid: RadialTimeGrid,
    *,
    omega_reference: float,
    amplitude: float = 2.0,
    tau_center: Optional[float] = None,
    cosh_scale: Optional[float] = None,  # optional time-stretch factor
    envelope_width: Optional[float] = None,
    flip_sign: bool = False,
    omega_tilde: Optional[float] = None,  # if provided apply e^{±(ω̃-ω)τ} tilt
    center_at_cloud: bool = False,
    decrease_towards_zero: bool = False,
    phase: float = 0.0,  # phase shift for complex mode: xi_complex *= exp(1j * phase)
) -> AnsatzResult:
    """
    Build the initial ansatz directly in the (y, ybar) variables, following the
    structure of Eq. (4.4):

        y_cl(r, τ)    ≃ [ y_Q(r) + A_- ( ξ_- e^{-iγτ} + ξ_-^* e^{+iγτ} ) ] e^{+(ω̃-ω)τ}
        ybar_cl(r, τ) ≃ [ y_Q(r) + A_- ( ξ_- e^{+iγτ} + ξ_-^* e^{-iγτ} ) ] e^{-(ω̃-ω)τ}

    with:
      - y_Q(r) the Q-cloud-like background in y-variables,
      - ξ_-(r) the complex unstable mode,
      - γ = mode.gamma,
      - ω̃ = omega_tilde when provided (otherwise no tilt is applied).

    The complex fields φ, φ̄ are reconstructed afterwards via phi_from_y / phi_from_ybar.
    """

    r_grid = grid.r          # (Nr,)
    tau = grid.tau           # (Ntau,)

    # --- Background y_Q(r)
    # |φ_Q(r)| is provided by the profile; interpolate onto the grid
    phi_base = np.interp(r_grid, profile.r, profile.phi_abs)

    # For y = r e^{-ωτ} φ we take the τ-independent background at τ = 0:
    yQ_r = r_grid * phi_base
    yQ_2D = yQ_r[:, None]  # broadcast across τ

    # --- Complex unstable mode ξ_-(r)
    xi_complex = np.interp(
        r_grid,
        mode.r,
        mode.xi_real + 1j * mode.xi_imag,
    )
    # convert χ fluctuations to φ fluctuations
    xi_complex /= np.sqrt(2.0)

    if flip_sign:
        xi_complex = -xi_complex

    # Apply phase shift: xi_complex *= exp(1j * phase)
    if phase != 0.0:
        xi_complex *= np.exp(1j * phase)

    xi_2D = xi_complex[:, None]

    # --- Time dependence e^{± i γ τ} (with optional rescaling)
    if tau_center is None:
        if center_at_cloud:
            tau_center_eff = float(grid.tau.min())
        else:
            tau_center_eff = 0.0
    else:
        tau_center_eff = tau_center

    tau_shift = tau - tau_center_eff

    if cosh_scale is not None and cosh_scale != 0.0:
        # act as a time stretch: τ_eff = (τ - τ_center)/cosh_scale
        tau_eff = tau_shift / cosh_scale
    else:
        tau_eff = tau_shift

    tau_eff_2D = tau_eff[None, :]

    gamma = mode.gamma

    phase_m = np.exp(-1j * gamma * tau_eff_2D)
    phase_p = np.exp(+1j * gamma * tau_eff_2D)

    term_osc_y    = xi_2D * phase_m + np.conj(xi_2D) * phase_p
    term_osc_ybar = xi_2D * phase_p + np.conj(xi_2D) * phase_m

    # --- Optional tilt e^{± (ω̃ - ω) τ_phys}
    if decrease_towards_zero:
        tau_phys = (-tau_shift)[None, :]
    else:
        tau_phys = tau_shift[None, :]
    if omega_tilde is not None:
        delta_omega = omega_tilde - omega_reference
        tilt_plus = np.exp(delta_omega * tau_phys)
        tilt_minus = np.exp(-delta_omega * tau_phys)
    else:
        tilt_plus = 1.0
        tilt_minus = 1.0

    # --- Optional Gaussian envelope in τ
    if envelope_width is not None and envelope_width > 0.0:
        envelope = np.exp(-(tau_shift / envelope_width) ** 2)[None, :]
        term_osc_y *= envelope
        term_osc_ybar *= envelope

    # --- Final ansatz in y, ybar
    y = (yQ_2D + amplitude * term_osc_y) * tilt_plus
    ybar = (yQ_2D + amplitude * term_osc_ybar) * tilt_minus

    # --- Reconstruct φ, φ̄
    phi = phi_from_y(y, grid, omega_reference)
    phibar = phi_from_ybar(ybar, grid, omega_reference)

    return AnsatzResult(phi=phi, phibar=phibar, y=y, ybar=ybar)


def build_custom_rho_ansatz_qball2d(
    grid: RadialTimeGrid,
    omega: float,
    profile: QBallProfile,
    *,
    sigma_r: float = 2.5,
    tau_center: Optional[float] = None,
    tau_width: float = 4.0,
    amplitude_scale: float = 0.5,
) -> AnsatzResult:
    """Separable rho ansatz used for exploratory visualisations."""

    r = grid.r[:, None]
    tau = grid.tau[None, :]

    if tau_center is None:
        tau_center = float(grid.tau.min())

    phi_cloud = np.interp(grid.r, profile.r, profile.phi_abs)[:, None]
    amplitude = amplitude_scale * phi_cloud

    envelope_r = np.exp(-0.5 * (r / sigma_r) ** 2)
    envelope_tau = 0.5 * (1.0 - np.tanh((tau - tau_center) / tau_width))
    rho = amplitude * envelope_r * envelope_tau

    phi = rho.astype(complex)
    phibar = np.conjugate(phi)

    y = r * np.exp(-omega * tau) * phi
    ybar = r * np.exp(+omega * tau) * phibar

    return AnsatzResult(phi=phi, phibar=phibar, y=y, ybar=ybar)


def build_qball_like_plateau_ansatz(
    grid: RadialTimeGrid,
    omega: float,
    profile: QBallProfile,
    *,
    tau_center: float,
    tau_width: float,
    amp_scale: float = 1.0,
    phase: float = 0.0,
) -> AnsatzResult:
    """
    Build a Q-ball-like plateau ansatz with separable structure: φ(r,τ) = ρ(r) * f(τ).
    
    The time function f(τ) transitions from plateau (f≈1) at τ << τ_center to vacuum (f≈0)
    at τ >> τ_center, modeling the metastable plateau → decay structure:
        f(τ) = 0.5 * (1 - tanh((τ - τ_center) / tau_width))
    
    The radial profile ρ(r) is taken from the provided QBallProfile.
    
    Parameters
    ----------
    grid
        Radial-time grid for the 2D solver.
    omega
        Frequency parameter for reconstructing y/ybar from φ.
    profile
        QBallProfile providing the radial profile ρ(r) = |φ(r)|.
    tau_center
        Center of the transition in Euclidean time (typically negative, e.g., -10 to -20).
    tau_width
        Width of the transition (typically 2-5).
    amp_scale
        Amplitude scaling factor for the profile (default 1.0).
    phase
        Optional phase factor: φ = ρ * exp(i*phase) (default 0.0, real field).
    
    Returns
    -------
    AnsatzResult
        Contains phi, phibar, y, ybar fields with the plateau ansatz structure.
    """
    r_grid = grid.r  # (Nr,)
    tau = grid.tau   # (Ntau,)
    
    # Interpolate radial profile onto grid
    rho_r = np.interp(r_grid, profile.r, profile.phi_abs)  # (Nr,)
    
    # Apply amplitude scaling
    rho_r = amp_scale * rho_r
    
    # Time function: f(τ) = 0.5 * (1 - tanh((τ - τ_center) / tau_width))
    # f ≈ 1 at τ << τ_center, f ≈ 0 at τ >> τ_center
    tau_2d = tau[None, :]  # (1, Ntau)
    f_tau = 0.5 * (1.0 - np.tanh((tau_2d - tau_center) / tau_width))  # (1, Ntau)
    
    # Build separable ansatz: φ(r,τ) = ρ(r) * f(τ) * exp(i*phase)
    rho_2d = rho_r[:, None]  # (Nr, 1)
    phi = (rho_2d * f_tau).astype(complex)  # (Nr, Ntau)
    
    # Apply phase if non-zero
    if phase != 0.0:
        phi = phi * np.exp(1j * phase)
    
    # Conjugate field
    phibar = np.conjugate(phi)
    
    # Reconstruct y, ybar from φ, φ̄ using solver conventions:
    # y = r * exp(-omega * tau) * φ
    # ybar = r * exp(+omega * tau) * φ̄
    r_2d = r_grid[:, None]  # (Nr, 1)
    tau_2d_expand = tau[None, :]  # (1, Ntau)
    
    y = r_2d * np.exp(-omega * tau_2d_expand) * phi
    ybar = r_2d * np.exp(+omega * tau_2d_expand) * phibar
    
    return AnsatzResult(phi=phi, phibar=phibar, y=y, ybar=ybar)


def build_qball_escape_ansatz(
    qball_profile: QBallProfile,
    mode: UnstableMode,
    grid: RadialTimeGrid,
    *,
    omega_reference: float,
    # gating function parameters:
    a_plateau: float = 0.35,  # target ρ(τ≈0) ≈ 0.5 in paper units
    tau_transition: float = -15.0,  # τ_* (negative)
    tau_width: float = 3.0,  # Δτ
    # optional localized negative-mode "kick":
    kick_amp: float = 0.0,  # set >0 to enable
    kick_theta: float = 0.0,  # phase
    kick_tau_width: float = 6.0,  # envelope width
    cosh_scale: Optional[float] = None,
    flip_sign: bool = False,
    omega_tilde: Optional[float] = None,
    decrease_towards_zero: bool = False,
) -> AnsatzResult:
    """
    Build Q-ball escape ansatz with gating function g(τ) that transitions from ~1
    at early times to a_plateau near τ≈0.
    
    Structure:
      phi_base = interp(grid.r, qball_profile.r, qball_profile.phi_abs)
      yQ_r = grid.r * phi_base  # τ-independent base in y
      yQ = yQ_r[:, None]  # broadcast in τ
      
      g(τ) = a_plateau + (1-a_plateau) * 0.5*(1 - tanh((τ - τ_trans)/w))
      y_gate = yQ * g[None, :]
      ybar_gate = yQ * g[None, :]
      
      Then add unstable-mode kick if kick_amp > 0.
    
    Parameters
    ----------
    qball_profile
        QBallProfile providing the Q-ball radial profile.
    mode
        UnstableMode for optional localized kick.
    grid
        Radial-time grid.
    omega_reference
        Reference frequency for y/ybar reconstruction.
    a_plateau
        Plateau value at τ≈0 (target ~0.3-0.4 to get ρ≈0.5 in paper units).
    tau_transition
        Center of transition in τ (negative, typically -15).
    tau_width
        Width of transition (typically 2-4).
    kick_amp
        Amplitude of unstable-mode kick (0 = disabled).
    kick_theta
        Phase for unstable-mode kick.
    kick_tau_width
        Width of Gaussian envelope for unstable-mode kick.
    cosh_scale
        Optional time-stretch for unstable-mode.
    flip_sign
        Flip sign of unstable mode.
    omega_tilde
        Optional tilt frequency.
    decrease_towards_zero
        Apply decrease_towards_zero tilt.
    
    Returns
    -------
    AnsatzResult
        Contains phi, phibar, y, ybar with Q-ball escape structure.
    """
    r_grid = grid.r  # (Nr,)
    tau = grid.tau   # (Ntau,)
    
    # --- Base Q-ball profile in y-variables
    phi_base = np.interp(r_grid, qball_profile.r, qball_profile.phi_abs)  # (Nr,)
    yQ_r = r_grid * phi_base  # τ-independent base in y
    yQ = yQ_r[:, None]  # broadcast in τ: (Nr, 1)
    
    # --- Gating function g(τ) = a_plateau + (1-a_plateau) * 0.5*(1 - tanh((τ - τ_trans)/w))
    # g ≈ 1 at early times, g ≈ a_plateau near τ≈0
    g = a_plateau + (1.0 - a_plateau) * 0.5 * (1.0 - np.tanh((tau - tau_transition) / tau_width))  # (Ntau,)
    
    # --- Build gated y, ybar
    y_gate = yQ * g[None, :]  # (Nr, Ntau)
    ybar_gate = yQ * g[None, :]  # (Nr, Ntau)
    
    # Convert to complex
    y = y_gate.astype(complex)
    ybar = ybar_gate.astype(complex)
    
    # --- Optional unstable-mode kick
    if kick_amp != 0.0:
        # Interpolate complex mode xi(r)
        xi_complex = np.interp(
            r_grid,
            mode.r,
            mode.xi_real + 1j * mode.xi_imag,
        )
        # convert χ fluctuations to φ fluctuations
        xi_complex /= np.sqrt(2.0)
        
        if flip_sign:
            xi_complex = -xi_complex
        
        # Apply phase kick_theta
        xi_complex *= np.exp(1j * kick_theta)
        
        xi_2D = xi_complex[:, None]  # (Nr, 1)
        
        # Time shift relative to tau_transition
        tau_shift = tau - tau_transition
        
        # Optional cosh scaling
        if cosh_scale is not None and cosh_scale != 0.0:
            tau_eff = tau_shift / cosh_scale
        else:
            tau_eff = tau_shift
        
        tau_eff_2D = tau_eff[None, :]  # (1, Ntau)
        
        gamma = mode.gamma
        
        # Oscillatory terms
        phase_m = np.exp(-1j * gamma * tau_eff_2D)
        phase_p = np.exp(+1j * gamma * tau_eff_2D)
        
        term_y = xi_2D * phase_m + np.conj(xi_2D) * phase_p
        term_ybar = xi_2D * phase_p + np.conj(xi_2D) * phase_m
        
        # Gaussian envelope localized around tau_transition
        env = np.exp(-((tau_shift / kick_tau_width) ** 2))[None, :]  # (1, Ntau)
        
        # Add to gated background
        y += kick_amp * term_y * env
        ybar += kick_amp * term_ybar * env
    
    # --- Optional tilt e^{± (ω̃ - ω) τ_phys}
    if omega_tilde is not None:
        if decrease_towards_zero:
            tau_phys = (-(tau - 0.0))[None, :]
        else:
            tau_phys = (tau - 0.0)[None, :]
        delta_omega = omega_tilde - omega_reference
        tilt_plus = np.exp(delta_omega * tau_phys)
        tilt_minus = np.exp(-delta_omega * tau_phys)
        y *= tilt_plus
        ybar *= tilt_minus
    
    # --- Reconstruct φ, φ̄ from y, ybar
    phi = phi_from_y(y, grid, omega_reference)
    phibar = phi_from_ybar(ybar, grid, omega_reference)
    
    return AnsatzResult(phi=phi, phibar=phibar, y=y, ybar=ybar)


__all__ = [
    "AnsatzResult",
    "build_negative_mode_ansatz",
    "build_custom_rho_ansatz_qball2d",
    "build_qball_like_plateau_ansatz",
    "build_qball_escape_ansatz",
]

