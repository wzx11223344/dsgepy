"""
Log-Linearization Engine
========================

Numerical log-linearization of DSGE model equilibrium conditions around
the deterministic steady state.

**Mathematical Framework**

Consider a system of :math:`n` equilibrium conditions:

.. math::

    f_i(X_t, X_{t-1}, E_t[X_{t+1}], \\varepsilon_t) = 0, \\quad i = 1, \\ldots, n

where :math:`X_t` is the :math:`n`-vector of model variables and
:math:`\\varepsilon_t` is the vector of structural shocks.

A **first-order Taylor approximation** around the steady state
:math:`\\bar{X}` (with :math:`\\bar{\\varepsilon} = 0`) yields:

.. math::

    \\mathbf{A} \\cdot E_t[\\hat{x}_{t+1}] +
    \\mathbf{B} \\cdot \\hat{x}_t +
    \\mathbf{C} \\cdot \\hat{x}_{t-1} +
    \\mathbf{D} \\cdot \\varepsilon_t = \\mathbf{0}

where :math:`\\hat{x}_t = \\ln X_t - \\ln \\bar{X}` denotes log-deviations
from steady state, and:

- :math:`\\mathbf{A}_{ij} = \\partial f_i / \\partial E_t[X_{t+1,j}]`
- :math:`\\mathbf{B}_{ij} = \\partial f_i / \\partial X_{t,j}`
- :math:`\\mathbf{C}_{ij} = \\partial f_i / \\partial X_{t-1,j}`
- :math:`\\mathbf{D}_{ij} = \\partial f_i / \\partial \\varepsilon_{t,j}`

All partial derivatives are evaluated at the steady state.

**Implementation**

Since we use numerical differentiation (central differences), the Jacobians
are computed by perturbing each variable individually and evaluating the
equilibrium conditions. For log-linearization:

.. math::

    X_t = \\bar{X} \\cdot \\exp(h)

where :math:`h` is the perturbation step size.

Functions
---------
log_linearize           : Symbolic-compatible log-linearization
first_order_approx       : First-order Taylor approximation
compute_jacobians        : Main method — returns A, B, C, D matrices
"""

import numpy as np


# =============================================================================
# Numerical Log-Linearization
# =============================================================================

def log_linearize(f, ss, variables, parameters):
    """
    Log-linearize a function around the steady state using numerical
    differentiation (central differences).

    For a function :math:`f(X)`, the log-linearized version computes
    :math:`\\partial f / \\partial \\ln X` evaluated at :math:`\\bar{X}`:

    .. math::

        \\frac{\\partial f}{\\partial \\ln X_j} \\approx
        \\frac{f(\\bar{X} e^{h e_j}) - f(\\bar{X} e^{-h e_j})}{2h}

    Parameters
    ----------
    f : callable
        The function to linearize. Should accept an array of variables
        and return a scalar residual.
    ss : dict or ndarray
        Steady state values.
    variables : list of str
        Variable names (for output labeling).
    parameters : dict
        Model parameters.

    Returns
    -------
    jacobian : ndarray
        Gradient vector evaluated at steady state.
    """
    n = len(variables)
    if isinstance(ss, dict):
        ss_vec = np.array([ss[v] for v in variables])
    else:
        ss_vec = np.asarray(ss)

    h = 1e-6  # perturbation step size
    jacobian = np.zeros(n)

    for j in range(n):
        # Forward perturbation
        x_plus = ss_vec.copy()
        x_plus[j] = ss_vec[j] * np.exp(h)
        f_plus = f(x_plus, ss_vec, x_plus, np.zeros(1))

        # Backward perturbation
        x_minus = ss_vec.copy()
        x_minus[j] = ss_vec[j] * np.exp(-h)
        f_minus = f(x_minus, ss_vec, x_minus, np.zeros(1))

        jacobian[j] = (f_plus - f_minus) / (2.0 * h)

    return jacobian


def first_order_approx(f_equations, steady_state, variables):
    """
    Compute first-order Taylor approximation of a system of equations.

    For a system :math:`F(X) = 0`, the first-order approximation is:

    .. math::

        F(\\bar{X}) + \\nabla F(\\bar{X}) \\cdot (X - \\bar{X}) = 0

    Parameters
    ----------
    f_equations : list of callable
        List of equilibrium condition functions. Each function takes
        ``(X_t, X_{t-1}, E_Xtp1, eps_t)`` and returns a scalar residual.
    steady_state : ndarray
        Steady state vector (:math:`\\bar{X}`).
    variables : list of str
        Variable names.

    Returns
    -------
    F_ss : ndarray (n,)
        Function values at steady state (should be near zero).
    J : ndarray (n, n)
        Jacobian matrix evaluated at steady state.
    """
    n = len(f_equations)
    m = len(steady_state)
    h = 1e-6

    F_ss = np.zeros(n)
    J = np.zeros((n, m))

    ss = steady_state.copy()

    for i, eq in enumerate(f_equations):
        # Evaluate at steady state (shocks = 0)
        eps_zero = np.zeros(1)
        F_ss[i] = eq(ss, ss, ss, eps_zero)

        for j in range(m):
            # Forward perturbation
            x_plus = ss.copy()
            dx = ss[j] * h if abs(ss[j]) > 1e-8 else h
            if abs(ss[j]) < 1e-12:
                dx = h
            x_plus[j] = ss[j] + dx

            # For log-variable perturbation, use multiplicative:
            if abs(ss[j]) > 1e-8:
                x_plus[j] = ss[j] * (1.0 + h)

            # Backward perturbation
            x_minus = ss.copy()
            if abs(ss[j]) > 1e-8:
                x_minus[j] = ss[j] * (1.0 - h)
            else:
                x_minus[j] = ss[j] - dx

            f_plus  = eq(x_plus, ss, x_plus, eps_zero)
            f_minus = eq(x_minus, ss, x_minus, eps_zero)

            delta = x_plus[j] - x_minus[j]
            if abs(delta) > 1e-15:
                J[i, j] = (f_plus - f_minus) / delta
            else:
                J[i, j] = 0.0

    return F_ss, J


# =============================================================================
# System Jacobians: A, B, C, D Matrices
# =============================================================================

def compute_jacobians(model, params=None):
    r"""
    Compute the log-linearized system Jacobians A, B, C, D.

    For each equilibrium condition :math:`f_i(X_t, X_{t-1}, E_t[X_{t+1}],
    \\varepsilon_t) = 0`, we compute:

    - **A matrix**: Derivatives w.r.t. :math:`E_t[X_{t+1}]` (forward-looking)
    - **B matrix**: Derivatives w.r.t. :math:`X_t` (contemporaneous)
    - **C matrix**: Derivatives w.r.t. :math:`X_{t-1}` (backward-looking)
    - **D matrix**: Derivatives w.r.t. :math:`\\varepsilon_t` (shocks)

    All derivatives are log-derivatives evaluated at the steady state:

    .. math::

        A_{ij} = \\left.\\frac{\\partial f_i}{\\partial \\ln E_t[X_{t+1,j}]}
        \\right|_{SS}

    .. math::

        B_{ij} = \\left.\\frac{\\partial f_i}{\\partial \\ln X_{t,j}}
        \\right|_{SS}

    .. math::

        C_{ij} = \\left.\\frac{\\partial f_i}{\\partial \\ln X_{t-1,j}}
        \\right|_{SS}

    .. math::

        D_{ij} = \\left.\\frac{\\partial f_i}{\\partial \\varepsilon_{t,j}}
        \\right|_{SS}

    After obtaining the Jacobians, we scale by steady-state values to
    convert from level derivatives to log-deviation derivatives:

    .. math::

        \\frac{\\partial f}{\\partial \\ln X} =
        \\frac{\\partial f}{\\partial X} \\cdot \\bar{X}

    Parameters
    ----------
    model : RBCModel or NewKeynesianModel or NKModelMedium
        A DSGE model instance with ``equilibrium_conditions()``,
        ``steady_state()``, and ``shock_matrix()`` methods.
    params : dict, optional
        Model parameters. If None, uses ``model.params``.

    Returns
    -------
    A : ndarray (n_vars, n_vars)
        Jacobian w.r.t. expected future variables.
    B : ndarray (n_vars, n_vars)
        Jacobian w.r.t. current variables.
    C : ndarray (n_vars, n_vars)
        Jacobian w.r.t. lagged variables.
    D : ndarray (n_vars, n_shocks)
        Jacobian w.r.t. structural shocks (or shock matrix).
    """
    if params is None:
        params = model.params

    # Get model components
    equations = model.equilibrium_conditions(params)
    eq_funcs = [eq[0] for eq in equations]
    ss_dict = model.steady_state(params)
    ss_vec = np.array([ss_dict[v] for v in model.variable_names])

    n = model.n_vars
    n_shocks = model.n_shocks
    h = 1e-6  # step size for numerical differentiation

    A = np.zeros((n, n))
    B = np.zeros((n, n))
    C = np.zeros((n, n))
    D_mat = np.zeros((n, n_shocks))

    ss_abs = np.abs(ss_vec)
    ss_abs[ss_abs < 1e-10] = 1.0  # avoid division by zero

    eps_zero = np.zeros(n_shocks)

    for i, eq in enumerate(eq_funcs):
        for j in range(n):
            # ----------
            # A matrix: derivative w.r.t. E_t[X_{t+1,j}]
            # ----------
            # Perturb forward (expected future)
            x_fwd_plus = ss_vec.copy()
            perturb = ss_vec[j] * h if abs(ss_vec[j]) > 1e-8 else h
            x_fwd_plus[j] = ss_vec[j] + perturb

            x_fwd_minus = ss_vec.copy()
            x_fwd_minus[j] = ss_vec[j] - perturb if abs(ss_vec[j]) > 1e-8 else ss_vec[j] - h

            f_fwd_plus  = eq(ss_vec, ss_vec, x_fwd_plus, eps_zero)
            f_fwd_minus = eq(ss_vec, ss_vec, x_fwd_minus, eps_zero)

            delta_fwd = x_fwd_plus[j] - x_fwd_minus[j]
            if abs(delta_fwd) > 1e-15:
                A[i, j] = (f_fwd_plus - f_fwd_minus) / delta_fwd
            else:
                A[i, j] = 0.0

            # ----------
            # B matrix: derivative w.r.t. X_t (contemporaneous)
            # ----------
            x_curr_plus = ss_vec.copy()
            perturb = ss_vec[j] * h if abs(ss_vec[j]) > 1e-8 else h
            x_curr_plus[j] = ss_vec[j] + perturb

            x_curr_minus = ss_vec.copy()
            x_curr_minus[j] = ss_vec[j] - perturb if abs(ss_vec[j]) > 1e-8 else ss_vec[j] - h

            f_curr_plus  = eq(x_curr_plus, ss_vec, ss_vec, eps_zero)
            f_curr_minus = eq(x_curr_minus, ss_vec, ss_vec, eps_zero)

            delta_curr = x_curr_plus[j] - x_curr_minus[j]
            if abs(delta_curr) > 1e-15:
                B[i, j] = (f_curr_plus - f_curr_minus) / delta_curr
            else:
                B[i, j] = 0.0

            # ----------
            # C matrix: derivative w.r.t. X_{t-1} (lagged)
            # ----------
            x_lag_plus = ss_vec.copy()
            perturb = ss_vec[j] * h if abs(ss_vec[j]) > 1e-8 else h
            x_lag_plus[j] = ss_vec[j] + perturb

            x_lag_minus = ss_vec.copy()
            x_lag_minus[j] = ss_vec[j] - perturb if abs(ss_vec[j]) > 1e-8 else ss_vec[j] - h

            f_lag_plus  = eq(ss_vec, x_lag_plus, ss_vec, eps_zero)
            f_lag_minus = eq(ss_vec, x_lag_minus, ss_vec, eps_zero)

            delta_lag = x_lag_plus[j] - x_lag_minus[j]
            if abs(delta_lag) > 1e-15:
                C[i, j] = (f_lag_plus - f_lag_minus) / delta_lag
            else:
                C[i, j] = 0.0

        # ----------
        # D matrix: derivative w.r.t. shocks (or use model's shock_matrix)
        # ----------
        for s in range(n_shocks):
            eps_plus = eps_zero.copy()
            eps_plus[s] = h
            eps_minus = eps_zero.copy()
            eps_minus[s] = -h

            f_eps_plus  = eq(ss_vec, ss_vec, ss_vec, eps_plus)
            f_eps_minus = eq(ss_vec, ss_vec, ss_vec, eps_minus)

            D_mat[i, s] = (f_eps_plus - f_eps_minus) / (2.0 * h)

    # Scale by steady-state values to convert from level to log-deviation
    # ∂f/∂log(X) = ∂f/∂X * X_bar
    for j in range(n):
        ss_j = ss_vec[j] if abs(ss_vec[j]) > 1e-12 else 1.0
        A[:, j] *= ss_j
        B[:, j] *= ss_j
        C[:, j] *= ss_j

    # If model provides a shock_matrix, use it for D (it's already in the right form)
    D_model = model.shock_matrix(params)
    if D_model is not None and np.any(D_model):
        D_mat = D_model

    return A, B, C, D_mat
