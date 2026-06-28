"""
Impulse Response Functions
==========================

Generate and compare impulse response functions (IRFs) from solved
DSGE models.

**Definition**

An impulse response function traces the dynamic response of model
variables to a one-time, one-standard-deviation structural shock:

.. math::

    \\mathbf{x}_t = \\mathbf{P} \\mathbf{x}_{t-1} +
    \\mathbf{Q} \\boldsymbol{\\varepsilon}_t

For a shock of size :math:`\\sigma_j` to shock :math:`j` at :math:`t=0`:

.. math::

    \\mathbf{x}_0 = \\mathbf{Q} \\cdot \\sigma_j \\mathbf{e}_j

    \\mathbf{x}_h = \\mathbf{P}^h \\mathbf{x}_0, \\quad h = 1, 2, \\ldots, H

Functions
---------
impulse_response         : IRF to a single shock
irf_comparison           : Compare IRFs across multiple models
variance_decomposition   : Forecast error variance decomposition
historical_decomposition : Historical shock decomposition
"""

import numpy as np


# =============================================================================
# Impulse Response Function
# =============================================================================

def impulse_response(P, Q, shock_idx=0, shock_std=1.0, horizon=40):
    r"""
    Compute the impulse response to a one-standard-deviation shock.

    The system evolves according to:

    .. math::

        \\mathbf{x}_t = \\mathbf{P} \\mathbf{x}_{t-1} +
        \\mathbf{Q} \\boldsymbol{\\varepsilon}_t

    with initial condition :math:`\\mathbf{x}_{-1} = \\mathbf{0}` and
    :math:`\\varepsilon_0 = \\sigma_j \\mathbf{e}_j` (all other
    :math:`\\varepsilon_t = 0` for :math:`t > 0`).

    Then:

    .. math::

        \\mathbf{x}_0 = \\mathbf{Q} \\cdot \\sigma_j \\mathbf{e}_j

        \\mathbf{x}_h = \\mathbf{P}^h \\mathbf{x}_0

    Parameters
    ----------
    P : ndarray (n, n)
        Transition matrix.
    Q : ndarray (n, k)
        Shock impact matrix.
    shock_idx : int
        Index of the shock to impulse (0-indexed).
    shock_std : float
        Standard deviation of the shock (scaling factor).
    horizon : int
        Number of periods to simulate forward.

    Returns
    -------
    irf : ndarray (horizon+1, n)
        Impulse response for each period and each variable.
        Row ``irf[h, :]`` gives the response at period ``h``.
    """
    n = P.shape[0]
    irf = np.zeros((horizon + 1, n))

    # Initial impact (period 0)
    shock = np.zeros(Q.shape[1])
    shock[shock_idx] = shock_std
    irf[0, :] = Q @ shock

    # Propagate forward
    for h in range(1, horizon + 1):
        irf[h, :] = P @ irf[h - 1, :]

    return irf


# =============================================================================
# IRF Comparison Across Models
# =============================================================================

def irf_comparison(models, shock_names, horizon=40):
    r"""
    Compare impulse response functions across multiple models.

    Parameters
    ----------
    models : list of tuple
        List of ``(model_name, P, Q, var_idx_dict)`` tuples.
        ``var_idx_dict`` maps variable names to indices in P/Q.
    shock_names : list of str
        Names of shocks to impulse.
    horizon : int
        IRF horizon.

    Returns
    -------
    results : dict
        Nested dict: ``results[model_name][shock_name]`` = ndarray (horizon+1, n_vars)
    """
    results = {}

    for model_name, P, Q, var_map in models:
        results[model_name] = {}
        for s_idx, shock_name in enumerate(shock_names):
            irf = impulse_response(P, Q, shock_idx=s_idx, horizon=horizon)
            results[model_name][shock_name] = irf

    return results


# =============================================================================
# Forecast Error Variance Decomposition
# =============================================================================

def variance_decomposition(P, Q, shock_std=None, horizon=40):
    r"""
    Compute forecast error variance decomposition (FEVD).

    For a VAR(1) process :math:`\mathbf{x}_t = \mathbf{P}\mathbf{x}_{t-1}
    + \mathbf{Q}\boldsymbol{\varepsilon}_t` with
    :math:`\boldsymbol{\varepsilon}_t \sim (0, \Sigma)`:

    The h-step-ahead forecast error is:

    .. math::

        \mathbf{x}_{t+h} - E_t[\mathbf{x}_{t+h}] =
        \sum_{s=0}^{h-1} \mathbf{P}^s \mathbf{Q} \boldsymbol{\varepsilon}_{t+h-s}

    The contribution of shock :math:`j` to the variance of variable :math:`i`
    at horizon :math:`h` is:

    .. math::

        \omega_{ij}(h) =
        \frac{\sum_{s=0}^{h-1} (\mathbf{P}^s \mathbf{Q} \mathbf{e}_j)^2_i}
        {\sum_{s=0}^{h-1} (\mathbf{P}^s \mathbf{Q} \Sigma \mathbf{Q}^T
        (\mathbf{P}^s)^T)_{ii}}

    Parameters
    ----------
    P : ndarray (n, n)
        Transition matrix.
    Q : ndarray (n, k)
        Shock impact matrix.
    shock_std : ndarray (k,), optional
        Standard deviations of shocks. If None, defaults to ones.
    horizon : int
        Maximum forecast horizon.

    Returns
    -------
    fevd : ndarray (horizon, n, k)
        Fraction of variance of variable i at horizon h
        attributable to shock j. ``fevd[h, i, j]``.
    """
    n = P.shape[0]
    k = Q.shape[1]

    if shock_std is None:
        shock_std = np.ones(k)

    # Shock covariance matrix (diagonal)
    Sigma = np.diag(shock_std ** 2)

    # Store cumulative squared contributions for each shock
    cum_sq = np.zeros((n, k))      # cumulative sum of squared contributions
    cum_total = np.zeros(n)        # cumulative total variance

    fevd = np.zeros((horizon, n, k))

    # Pre-compute P^s for each s
    P_power = np.eye(n)
    for h in range(horizon):
        # Contribution of shocks at this step
        contrib = P_power @ Q        # (n, k) = P^h * Q

        for j in range(k):
            sq_contrib = (contrib[:, j] * shock_std[j]) ** 2
            cum_sq[:, j] += sq_contrib

        # Total variance at this horizon
        total_var = P_power @ Q @ Sigma @ Q.T @ P_power.T
        cum_total += np.diag(total_var)

        # Compute FEVD
        for i in range(n):
            if cum_total[i] > 1e-15:
                fevd[h, i, :] = cum_sq[i, :] / cum_total[i]
            else:
                fevd[h, i, :] = 0.0

        P_power = P @ P_power

    return fevd


# =============================================================================
# Historical Decomposition
# =============================================================================

def historical_decomposition(P, Q, shocks, data):
    r"""
    Perform historical shock decomposition of observed data.

    Given a series of estimated shocks :math:`\{\boldsymbol{\varepsilon}_t\}`
    and observed data :math:`\{\mathbf{x}_t\}`, decompose each variable's
    path into the cumulative contribution of each structural shock.

    The contribution of shock :math:`j` to variable :math:`i` at time
    :math:`t` is:

    .. math::

        x_{it}^{(j)} = \sum_{s=0}^{t} (\mathbf{P}^s \mathbf{Q})_{ij}
        \cdot \varepsilon_{t-s,j}

    Parameters
    ----------
    P : ndarray (n, n)
        Transition matrix.
    Q : ndarray (n, k)
        Shock impact matrix.
    shocks : ndarray (T, k)
        Estimated structural shocks.
    data : ndarray (T, n), optional
        Observed data (used for validation). If None, only shock
        contributions are returned.

    Returns
    -------
    decomp : ndarray (T, n, k)
        ``decomp[t, i, j]`` = contribution of shock j to variable i at time t.
    """
    T = shocks.shape[0]
    n = P.shape[0]
    k = Q.shape[1]

    decomp = np.zeros((T, n, k))

    # Compute impulse response weights
    # weight[s, :, :] = P^s * Q
    max_horizon = min(T, 100)  # truncate for long series
    weights = np.zeros((max_horizon, n, k))
    weights[0] = Q
    P_power = np.eye(n)
    for s in range(1, max_horizon):
        P_power = P @ P_power
        weights[s] = P_power @ Q

    # Accumulate contributions
    for t in range(T):
        for s in range(min(t + 1, max_horizon)):
            for j in range(k):
                decomp[t, :, j] += weights[s, :, j] * shocks[t - s, j]

    return decomp
