"""
DSGEpy — Dynamic Stochastic General Equilibrium Modeling Toolkit.

A lightweight, pure-Python toolkit for building, solving, simulating, and
estimating DSGE models. Implements log-linearization, Blanchard-Kahn / Klein
solvers, impulse response functions, stochastic simulation, and Bayesian MCMC
estimation.

Modules
-------
models      : Pre-built DSGE model definitions (RBC, NK, medium-scale NK)
linearize   : Log-linearization engine (numerical Jacobian)
solution    : Linear rational expectations solvers (BK, Klein)
irf         : Impulse response functions, variance decomposition
simulation  : Stochastic simulation, moment matching, Kalman filter
estimation  : Bayesian MCMC estimation (Random Walk Metropolis-Hastings)
"""

from dsgepy.models import RBCModel, NewKeynesianModel, NKModelMedium
from dsgepy.linearize import (
    log_linearize,
    first_order_approx,
    compute_jacobians,
)
from dsgepy.solution import (
    blanchard_kahn,
    klein_solver,
    solve_model,
)
from dsgepy.irf import (
    impulse_response,
    irf_comparison,
    variance_decomposition,
    historical_decomposition,
)
from dsgepy.simulation import (
    simulate_model,
    compute_moments,
    moment_matching,
    filtered_variables,
)
from dsgepy.estimation import (
    bayesian_estimation,
    compute_marginal_likelihood,
    log_posterior,
)

__version__ = "0.1.0"
__all__ = [
    # models
    "RBCModel", "NewKeynesianModel", "NKModelMedium",
    # linearize
    "log_linearize", "first_order_approx", "compute_jacobians",
    # solution
    "blanchard_kahn", "klein_solver", "solve_model",
    # irf
    "impulse_response", "irf_comparison", "variance_decomposition",
    "historical_decomposition",
    # simulation
    "simulate_model", "compute_moments", "moment_matching",
    "filtered_variables",
    # estimation
    "bayesian_estimation", "compute_marginal_likelihood", "log_posterior",
]
