# DSGEpy

**Dynamic Stochastic General Equilibrium** modeling toolkit for Python.

DSGEpy provides a complete pipeline for building, solving, simulating, and estimating DSGE models. It implements standard solution methods (Blanchard-Kahn, Klein) and comes with pre-built RBC and New Keynesian models.

---

## Features

- **Pre-built Models** — RBC, 3-equation New Keynesian, medium-scale NK
- **Log-Linearization** — Symbolic-compatible and numerical Jacobian engines
- **Linear RE Solvers** — Blanchard-Kahn (1980), Klein (2000), Sims (2002)
- **Impulse Response Functions** — IRF generation, cross-model comparison, variance decomposition
- **Stochastic Simulation** — Model simulation, moment computation, Kalman filtering
- **Bayesian Estimation** — Random Walk Metropolis-Hastings with custom priors

## Installation

```bash
cd dsgepy
pip install -e .
```

## Quick Start

```python
import numpy as np
from dsgepy.models import RBCModel
from dsgepy.linearize import compute_jacobians
from dsgepy.solution import solve_model
from dsgepy.irf import impulse_response

# 1. Define parameters
params = {
    'beta': 0.99, 'alpha': 0.36, 'delta': 0.025,
    'chi': 1.5, 'rho_z': 0.95, 'sigma_z': 0.01,
}

# 2. Build model
rbc = RBCModel(params)

# 3. Compute steady state
ss = rbc.steady_state(params)
print(f"Steady state K/Y ratio: {ss['ky_ratio']:.3f}")

# 4. Log-linearize
A, B, C, D = compute_jacobians(rbc, params)

# 5. Solve with Blanchard-Kahn
result = solve_model(rbc, params)
P, Q = result['P'], result['Q']

# 6. Impulse response to TFP shock
irf = impulse_response(P, Q, shock_idx=0, horizon=40)
```

## Models

| Model | Equations | Key Features |
|-------|-----------|--------------|
| `RBCModel` | 5 equations | Cobb-Douglas production, AR(1) TFP shock |
| `NewKeynesianModel` | 5 equations | Dynamic IS, NK Phillips Curve, Taylor Rule |
| `NKModelMedium` | 10 equations | Wage/price rigidities, investment adjustment costs |

## Theory

DSGEpy implements the standard linear rational expectations solution framework:

$$\mathbf{A} \cdot E_t[\mathbf{x}_{t+1}] + \mathbf{B} \cdot \mathbf{x}_t + \mathbf{C} \cdot \mathbf{x}_{t-1} + \mathbf{D} \cdot \boldsymbol{\varepsilon}_t = \mathbf{0}$$

The Blanchard-Kahn method uses generalized Schur (QZ) decomposition to partition eigenvalues into stable and unstable subspaces, yielding the solution:

$$\mathbf{x}_t = \mathbf{P} \cdot \mathbf{x}_{t-1} + \mathbf{Q} \cdot \boldsymbol{\varepsilon}_t$$

## License

MIT License. See [LICENSE](LICENSE) for details.

## References

- Blanchard, O. and Kahn, C. (1980). "The Solution of Linear Difference Models under Rational Expectations." *Econometrica*, 48(5), 1305-1311.
- Klein, P. (2000). "Using the Generalized Schur Form to Solve a Multivariate Linear Rational Expectations Model." *Journal of Economic Dynamics and Control*, 24(10), 1405-1423.
- Sims, C. (2002). "Solving Linear Rational Expectations Models." *Computational Economics*, 20(1-2), 1-20.
- Schmitt-Grohe, S. and Uribe, M. (2004). "Solving Dynamic General Equilibrium Models Using a Second-Order Approximation to the Policy Function." *Journal of Economic Dynamics and Control*, 28(4), 755-775.
