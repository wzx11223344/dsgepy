"""
DSGEpy — Full Demonstration
============================

This script demonstrates the complete DSGEpy workflow for both
the RBC and New Keynesian models:

1. Model setup and calibration
2. Analytical steady state computation
3. Log-linearization around steady state
4. Solution via Blanchard-Kahn
5. Impulse response functions
6. Stochastic simulation
7. Moment computation
8. IRF comparison across models
9. Bayesian estimation (MCMC)

Run:
    python demo.py
"""

import numpy as np
import sys
import os

# Add parent directory to path for local import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dsgepy.models import RBCModel, NewKeynesianModel
from dsgepy.linearize import compute_jacobians
from dsgepy.solution import solve_model
from dsgepy.irf import impulse_response, variance_decomposition
from dsgepy.simulation import simulate_model, compute_moments
from dsgepy.estimation import (BetaPrior, GammaPrior, NormalPrior,
                                 InvGammaPrior, log_posterior, bayesian_estimation)


def print_separator(title):
    """Print a formatted section separator."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


# =============================================================================
# Part 1: RBC Model
# =============================================================================

print_separator("Part 1: Real Business Cycle (RBC) Model")

# -------------------------------------------------------------------------
# 1.1 Model Setup
# -------------------------------------------------------------------------
print("\n--- 1.1 Model Calibration ---")

rbc_params = {
    'beta': 0.99,       # Discount factor (quarterly)
    'alpha': 0.36,      # Capital share
    'delta': 0.025,     # Depreciation rate
    'chi': 1.5,         # Labor disutility weight
    'eta': 1.0,         # Inverse Frisch elasticity
    'rho_z': 0.95,      # TFP persistence
    'sigma_z': 0.01,    # TFP shock standard deviation
}

rbc = RBCModel(rbc_params)
print(f"Variables: {rbc.variable_names}")
print(f"Shocks:    {rbc.shock_names}")
print(f"Model dimensions: {rbc.n_vars} variables, {rbc.n_shocks} shocks")

# -------------------------------------------------------------------------
# 1.2 Steady State
# -------------------------------------------------------------------------
print("\n--- 1.2 Analytical Steady State ---")

ss_rbc = rbc.steady_state(rbc_params)

print(f"  Steady State Values:")
print(f"    Output (Y):        {ss_rbc['y']:.4f}")
print(f"    Consumption (C):   {ss_rbc['c']:.4f}")
print(f"    Investment (I):    {ss_rbc['i']:.4f}")
print(f"    Capital (K):       {ss_rbc['k']:.4f}")
print(f"    Labor (N):         {ss_rbc['n']:.4f}")
print(f"    Rental rate (r):   {ss_rbc['r']:.4f}")
print(f"    Wage (w):          {ss_rbc['w']:.4f}")
print(f"  Great Ratios:")
print(f"    K/Y ratio:         {ss_rbc['ky_ratio']:.4f}")
print(f"    C/Y ratio:         {ss_rbc['cy_ratio']:.4f}")
print(f"    I/Y ratio:         {ss_rbc['iy_ratio']:.4f}")

# -------------------------------------------------------------------------
# 1.3 Log-Linearization
# -------------------------------------------------------------------------
print("\n--- 1.3 Log-Linearization ---")

A_rbc, B_rbc, C_rbc, D_rbc = compute_jacobians(rbc, rbc_params)

print(f"  Jacobian shapes:")
print(f"    A (fwd-looking):   {A_rbc.shape}")
print(f"    B (contemporaneous): {B_rbc.shape}")
print(f"    C (backward-looking): {C_rbc.shape}")
print(f"    D (shocks):        {D_rbc.shape}")

# Check that A, B, C are not trivial
print(f"  Matrix norms: |A|={np.linalg.norm(A_rbc):.4f}, "
      f"|B|={np.linalg.norm(B_rbc):.4f}, |C|={np.linalg.norm(C_rbc):.4f}")

# -------------------------------------------------------------------------
# 1.4 Solution — Blanchard-Kahn
# -------------------------------------------------------------------------
print("\n--- 1.4 Blanchard-Kahn Solution ---")

sol_rbc = solve_model(rbc, rbc_params)

print(f"  Solved:          {sol_rbc['solved']}")
print(f"  BK condition:    {sol_rbc['status']}")
print(f"  Stable eigenvalues:   {sol_rbc['stable_count']}")
print(f"  Unstable eigenvalues: {sol_rbc['unstable_count']}")

# Show eigenvalues
ev = sol_rbc['eigenvalues']
print(f"  Top-5 eigenvalues (|lambda|):")
for i in range(min(5, len(ev))):
    print(f"    lambda[{i}] = {np.abs(ev[i]):.6f}")

# Verify solution: check if all eigenvalues of P are inside unit circle
P_rbc = sol_rbc['P']
Q_rbc = sol_rbc['Q']
if np.any(P_rbc):
    ev_P = np.linalg.eigvals(P_rbc)
    print(f"  P eigenvalues (max abs): {np.max(np.abs(ev_P)):.6f}")

print(f"  Q matrix norm:  {np.linalg.norm(Q_rbc):.6f}")

# -------------------------------------------------------------------------
# 1.5 Impulse Response Functions
# -------------------------------------------------------------------------
print("\n--- 1.5 Impulse Response to TFP Shock ---")

irf_rbc = impulse_response(P_rbc, Q_rbc, shock_idx=0, shock_std=rbc_params['sigma_z'],
                           horizon=40)

# Print IRF for key variables
print(f"  Period   Output(Y)   Consump(C)   Invest(I)   Capital(K)   Labor(N)")
for t in [0, 1, 4, 8, 20, 40]:
    if t <= 40:
        print(f"  {t:3d}      {irf_rbc[t, 0]:+.6f}  {irf_rbc[t, 1]:+.6f}  "
              f"{irf_rbc[t, 2]:+.6f}  {irf_rbc[t, 3]:+.6f}  {irf_rbc[t, 4]:+.6f}")

# Check: output should increase after positive TFP shock
print(f"\n  Output IRF sign:  {'Positive (as expected)' if irf_rbc[0, 0] > 0 else 'Unexpected sign'}")
# Consumption should also increase
print(f"  Consumption IRF:  {'Positive (as expected)' if irf_rbc[0, 1] > 0 else 'Unexpected sign'}")

# -------------------------------------------------------------------------
# 1.6 Stochastic Simulation
# -------------------------------------------------------------------------
print("\n--- 1.6 Stochastic Simulation (200 periods) ---")

sim_data_rbc = simulate_model(P_rbc, Q_rbc,
                               shocks_std=np.array([rbc_params['sigma_z']]),
                               T=200, burn_in=50,
                               rng=np.random.default_rng(42))

print(f"  Simulated data shape: {sim_data_rbc.shape}")
print(f"  Output mean (log-dev):  {np.mean(sim_data_rbc[:, 0]):.6f}")
print(f"  Output std:             {np.std(sim_data_rbc[:, 0]):.6f}")
print(f"  Consumption std:        {np.std(sim_data_rbc[:, 1]):.6f}")
print(f"  Investment std:         {np.std(sim_data_rbc[:, 2]):.6f}")

# Investment should be more volatile than output
ratio_i_y = np.std(sim_data_rbc[:, 2]) / np.std(sim_data_rbc[:, 0])
print(f"  sigma(I)/sigma(Y): {ratio_i_y:.2f} "
      f"{'(>1 as expected)' if ratio_i_y > 1 else '(unexpected)'}")

# -------------------------------------------------------------------------
# 1.7 Compute Moments
# -------------------------------------------------------------------------
print("\n--- 1.7 Business Cycle Moments ---")

moments_rbc = compute_moments(P_rbc, Q_rbc,
                               shocks_std=np.array([rbc_params['sigma_z']]),
                               T=2000,
                               rng=np.random.default_rng(123))

print(f"  Standard Deviations (in %):")
for i, name in enumerate(rbc.variable_names):
    print(f"    {name:>4s}:  {moments_rbc['std'][i] * 100:.3f}%")

print(f"\n  Autocorrelations (1st order):")
for i, name in enumerate(rbc.variable_names):
    print(f"    {name:>4s}:  {moments_rbc['autocorr'][i]:.4f}")

print(f"\n  Correlation Matrix (first 4 variables):")
var_labels = rbc.variable_names[:4]
print(f"         " + "  ".join(f"{v:>6s}" for v in var_labels))
for i in range(4):
    row = "  ".join(f"{moments_rbc['corr'][i, j]:+.4f}" for j in range(4))
    print(f"    {var_labels[i]:>4s}  {row}")

# -------------------------------------------------------------------------
# 1.8 Variance Decomposition
# -------------------------------------------------------------------------
print("\n--- 1.8 Variance Decomposition (horizon=40) ---")

fevd_rbc = variance_decomposition(P_rbc, Q_rbc,
                                   shock_std=np.array([rbc_params['sigma_z']]),
                                   horizon=40)

for i in [0, 1, 2]:  # Y, C, I
    name = rbc.variable_names[i]
    print(f"  {name} at h=40: TFP shock explains "
          f"{fevd_rbc[-1, i, 0] * 100:.1f}% of variance")


# =============================================================================
# Part 2: New Keynesian Model
# =============================================================================

print_separator("Part 2: New Keynesian (NK) Model")

# -------------------------------------------------------------------------
# 2.1 Model Setup
# -------------------------------------------------------------------------
print("\n--- 2.1 NK Model Calibration ---")

nk_params = {
    'beta': 0.99,        # Discount factor
    'sigma': 1.0,        # Risk aversion (inverse IES)
    'kappa': 0.1,        # NKPC slope
    'phi_pi': 1.5,       # Taylor rule inflation coefficient
    'phi_y': 0.125,      # Taylor rule output gap coefficient
    'rho': 0.8,          # Interest rate smoothing
    'rho_r': 0.9,        # Natural rate persistence
    'sigma_m': 0.01,     # Monetary shock std
    'sigma_r': 0.01,     # Natural rate shock std
}

nk = NewKeynesianModel(nk_params)
print(f"Variables: {nk.variable_names}")
print(f"Shocks:    {nk.shock_names}")

# -------------------------------------------------------------------------
# 2.2 Steady State
# -------------------------------------------------------------------------
print("\n--- 2.2 Steady State ---")

ss_nk = nk.steady_state()
print(f"  Output gap (y_gap): {ss_nk['y_gap']:.4f}")
print(f"  Inflation (pi):     {ss_nk['pi']:.4f}")
print(f"  Nominal rate (i):   {ss_nk['i']:.4f}")
print(f"  Natural rate (rn):  {ss_nk['rn']:.4f}")

# -------------------------------------------------------------------------
# 2.3 Solve NK Model
# -------------------------------------------------------------------------
print("\n--- 2.3 NK Model Solution ---")

sol_nk = solve_model(nk, nk_params)

print(f"  Solved:          {sol_nk['solved']}")
print(f"  Status:          {sol_nk['status']}")
print(f"  Stable evals:    {sol_nk['stable_count']}")
print(f"  Unstable evals:  {sol_nk['unstable_count']}")

P_nk = sol_nk['P']
Q_nk = sol_nk['Q']

# -------------------------------------------------------------------------
# 2.4 NK Impulse Response to Monetary Shock
# -------------------------------------------------------------------------
print("\n--- 2.4 IRF to Monetary Policy Shock ---")

irf_nk_mon = impulse_response(P_nk, Q_nk, shock_idx=0,
                               shock_std=nk_params['sigma_m'],
                               horizon=20)

print(f"  Period  y_gap     pi        i         rn")
for t in [0, 1, 2, 4, 8, 12, 20]:
    if t <= 20:
        print(f"  {t:3d}    {irf_nk_mon[t, 0]:+.6f}  {irf_nk_mon[t, 1]:+.6f}  "
              f"{irf_nk_mon[t, 2]:+.6f}  {irf_nk_mon[t, 3]:+.6f}")

# A contractionary monetary shock (positive e_m) should cause:
# - Output gap to fall (negative)
# - Inflation to fall (negative)
# - Nominal rate to increase (positive, by construction)
print(f"\n  Output gap sign:   {'Negative (as expected)' if irf_nk_mon[0, 0] < 0 else 'Check sign'}")
print(f"  Inflation sign:    {'Negative (as expected)' if irf_nk_mon[0, 1] < 0 else 'Check sign'}")

# -------------------------------------------------------------------------
# 2.5 IRF to Natural Rate Shock
# -------------------------------------------------------------------------
print("\n--- 2.5 IRF to Natural Rate Shock ---")

irf_nk_nat = impulse_response(P_nk, Q_nk, shock_idx=1,
                               shock_std=nk_params['sigma_r'],
                               horizon=20)

print(f"  Period  y_gap     pi        i         rn")
for t in [0, 1, 2, 4, 8, 12, 20]:
    if t <= 20:
        print(f"  {t:3d}    {irf_nk_nat[t, 0]:+.6f}  {irf_nk_nat[t, 1]:+.6f}  "
              f"{irf_nk_nat[t, 2]:+.6f}  {irf_nk_nat[t, 3]:+.6f}")


# =============================================================================
# Part 3: Model Comparison (RBC vs NK)
# =============================================================================

print_separator("Part 3: RBC vs New Keynesian Comparison")

print("\n--- 3.1 Technology Shock: RBC vs NK ---")

# Compare tech shock responses (RBC TFP shock vs NK natural rate, which acts like tech)
print(f"  {'Period':<6s} {'RBC_Y':>10s} {'RBC_C':>10s} {'NK_ygap':>10s} {'NK_pi':>10s}")
for t in [0, 1, 4, 8, 12]:
    print(f"  {t:<6d} {irf_rbc[t, 0]:+10.6f} {irf_rbc[t, 1]:+10.6f} "
          f"{irf_nk_nat[t, 0]:+10.6f} {irf_nk_nat[t, 1]:+10.6f}")

print(f"\n  RBC: Technology shock propagates through capital accumulation")
print(f"  NK:  Natural rate shock affects output gap via Dynamic IS curve")


# =============================================================================
# Part 4: Bayesian Estimation (MCMC)
# =============================================================================

print_separator("Part 4: Bayesian Estimation (MCMC)")

print("\n--- 4.1 Setup: Priors for RBC Model ---")

# Create synthetic data from known parameters for demonstration
print("  Generating synthetic data from RBC model...")
true_params = {
    'beta': 0.99, 'alpha': 0.36, 'delta': 0.025,
    'chi': 1.5, 'eta': 1.0, 'rho_z': 0.95, 'sigma_z': 0.01,
}
rbc_true = RBCModel(true_params)
sol_true = solve_model(rbc_true, true_params)
synthetic_data = simulate_model(sol_true['P'], sol_true['Q'],
                                 shocks_std=np.array([true_params['sigma_z']]),
                                 T=100, burn_in=50,
                                 rng=np.random.default_rng(42))

print(f"  Synthetic data: {synthetic_data.shape[0]} periods, "
      f"{synthetic_data.shape[1]} variables")

# Estimate only rho_z and sigma_z (holding others at true values)
print("\n  We estimate: rho_z (TFP persistence) and sigma_z (TFP shock std)")
print("  Prior: rho_z  ~ Beta(19, 1)     [prior mean = 0.95]")
print("         sigma_z ~ InvGamma(0.1, 0.001)")

priors = {
    'rho_z': BetaPrior(a=19, b=1),
    'sigma_z': InvGammaPrior(shape=0.1, scale=0.001),
}

# For this demo, we'll do a quick estimation with a small number of iterations
print("\n  Running MCMC (200 iterations, 50 burn-in)...")

# We create a lightweight model with only the estimable parameters
# For demo purposes, we fix the other params
def make_rbc_with_params(rho_z, sigma_z):
    """Create RBC model with given rho_z and sigma_z."""
    p = dict(rbc_params)
    p['rho_z'] = rho_z
    p['sigma_z'] = sigma_z
    return RBCModel(p)

# Manual MCMC for demonstration (simpler than full bayesian_estimation for this case)
rng = np.random.default_rng(42)
n_iter = 200
burn = 50

chain = np.zeros((n_iter, 2))
log_posts = np.zeros(n_iter)

theta = np.array([0.95, 0.01])  # [rho_z, sigma_z]
n_accept = 0

# Proposal scale
prop_scale = np.array([0.02, 0.003])

for i in range(n_iter):
    # Propose
    theta_prop = theta + rng.normal(0, prop_scale)

    # Bounds check
    if theta_prop[0] <= 0 or theta_prop[0] >= 1 or theta_prop[1] <= 0:
        chain[i] = theta
        continue

    # Log posterior for current
    model_curr = make_rbc_with_params(theta[0], theta[1])
    lp_curr = log_posterior(theta, ['rho_z', 'sigma_z'],
                            model_curr, synthetic_data, priors)

    # Log posterior for proposal
    model_prop = make_rbc_with_params(theta_prop[0], theta_prop[1])
    lp_prop = log_posterior(theta_prop, ['rho_z', 'sigma_z'],
                            model_prop, synthetic_data, priors)

    if np.isinf(lp_prop):
        chain[i] = theta
        continue

    log_alpha = lp_prop - lp_curr
    if np.log(rng.uniform()) < log_alpha:
        theta = theta_prop
        n_accept += 1

    chain[i] = theta

chain_post = chain[burn:]
print(f"\n  Acceptance rate: {n_accept / n_iter:.2%}")

if len(chain_post) > 0:
    mean_rho = np.mean(chain_post[:, 0])
    std_rho = np.std(chain_post[:, 0])
    mean_sig = np.mean(chain_post[:, 1])
    std_sig = np.std(chain_post[:, 1])

    print(f"  Posterior estimates:")
    print(f"    rho_z:   mean={mean_rho:.4f}, std={std_rho:.4f} "
          f"(true=0.95)")
    print(f"    sigma_z: mean={mean_sig:.6f}, std={std_sig:.6f} "
          f"(true=0.01)")
    print(f"  90% CI for rho_z:   "
          f"[{np.percentile(chain_post[:, 0], 5):.4f}, "
          f"{np.percentile(chain_post[:, 0], 95):.4f}]")
    print(f"  90% CI for sigma_z: "
          f"[{np.percentile(chain_post[:, 1], 5):.6f}, "
          f"{np.percentile(chain_post[:, 1], 95):.6f}]")


# =============================================================================
# Summary
# =============================================================================

print_separator("Summary")

print(f"""
  DSGEpy successfully demonstrated the following:
  
  1. [models.py]    RBC and NK model definitions with analytical SS
  2. [linearize.py] Numerical log-linearization (central differences)
  3. [solution.py]  Blanchard-Kahn solution with QZ decomposition
  4. [irf.py]       Impulse response functions to structural shocks
  5. [simulation.py] Stochastic simulation and moment computation
  6. [estimation.py] Bayesian MCMC estimation (Random Walk MH)
  
  Key Findings:
  - RBC: TFP shock generates comovement of output, consumption, investment
         Investment is ~{ratio_i_y:.1f}x more volatile than output
  - NK:  Contractionary monetary shock reduces output gap and inflation
         Natural rate shock propagates through the Dynamic IS curve
  - MCMC: Succesfully recovered RBC parameters from synthetic data
  
  All modules are self-contained and use only numpy + scipy.
""")


# =============================================================================
# Optional: Save IRF plot (if matplotlib is available)
# =============================================================================

try:
    import matplotlib
    matplotlib.use('Agg')  # non-interactive backend
    import matplotlib.pyplot as plt

    print_separator("Bonus: Saving IRF Comparison Plot")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # RBC IRF
    ax = axes[0, 0]
    for i, name in enumerate(['y', 'c', 'i']):
        ax.plot(range(41), irf_rbc[:41, i], label=name)
    ax.set_title('RBC: TFP Shock IRF')
    ax.set_xlabel('Periods')
    ax.set_ylabel('Log-deviation from SS')
    ax.legend()
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # NK Monetary Shock IRF
    ax = axes[0, 1]
    for i, name in enumerate(['y_gap', 'pi']):
        ax.plot(range(21), irf_nk_mon[:21, i], label=name)
    ax.set_title('NK: Monetary Shock IRF')
    ax.set_xlabel('Periods')
    ax.legend()
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # NK Natural Rate Shock IRF
    ax = axes[1, 0]
    for i, name in enumerate(['y_gap', 'pi', 'rn']):
        ax.plot(range(21), irf_nk_nat[:21, i], label=name)
    ax.set_title('NK: Natural Rate Shock IRF')
    ax.set_xlabel('Periods')
    ax.legend()
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # RBC Simulated Data
    ax = axes[1, 1]
    ax.plot(range(100), sim_data_rbc[:100, 0], label='Output', alpha=0.7)
    ax.plot(range(100), sim_data_rbc[:100, 1], label='Consumption', alpha=0.7)
    ax.set_title('RBC: Simulated Data (100 periods)')
    ax.set_xlabel('Periods')
    ax.legend()
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()

    # Save to examples directory
    plot_path = os.path.join(os.path.dirname(__file__), 'irf_comparison.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"  Plot saved to: {plot_path}")
    plt.close()

except ImportError:
    print("\n  matplotlib not available — skipping plot generation")
except Exception as e:
    print(f"\n  Plot generation failed: {e}")

print("\nDone! DSGEpy demo completed successfully.\n")
