"""
Stochastic Simulation and Moment Matching
=========================================

Simulate solved DSGE models and compute business cycle moments.

**Simulation**

Given the solved model in reduced form:

.. math::

    \\mathbf{x}_t = \\mathbf{P} \\mathbf{x}_{t-1} +
    \\mathbf{Q} \\boldsymbol{\\varepsilon}_t,
    \\quad \\boldsymbol{\\varepsilon}_t \\sim \\mathcal{N}(\\mathbf{0}, \\Sigma)

we can simulate artificial time series by drawing random shocks.

**Moment Computation**

Common business cycle moments:
- Standard deviations: :math:`\\sigma(x_i)`
- Autocorrelations: :math:`\\rho(x_i, x_{i,t-1})`
- Cross-correlations: :math:`\\rho(x_i, x_j)`

**Moment Matching**

Compare model-implied moments against empirical data moments to assess
model fit:

.. math::

    d(\\theta) = \\mathbf{m}_{model}(\\theta) - \\mathbf{m}_{data}

Functions
---------
simulate_model       : Generate simulated data from solved model
compute_moments      : Compute standard deviations, correlations, autocorrelations
moment_matching       : Compare model moments vs empirical moments
filtered_variables    : Kalman filter for state estimation
"""

import numpy as np


# =============================================================================
# Model Simulation
# =============================================================================

def simulate_model(P, Q, shocks_std=None, T=200, n_simulations=1,
                   burn_in=50, rng=None):
    r"""
    Simulate the solved DSGE model.

    Simulates the process:

    .. math::

        \\mathbf{x}_t = \\mathbf{P} \\mathbf{x}_{t-1} +
        \\mathbf{Q} \\boldsymbol{\\varepsilon}_t,
        \\quad \\boldsymbol{\\varepsilon}_t \\sim
        \\mathcal{N}(\\mathbf{0}, \\Sigma)

    starting from :math:`\\mathbf{x}_0 = \\mathbf{0}` and discarding
    a burn-in period to eliminate transient effects.

    Parameters
    ----------
    P : ndarray (n, n)
        Transition matrix.
    Q : ndarray (n, k)
        Shock impact matrix.
    shocks_std : ndarray (k,), optional
        Standard deviations of structural shocks. If None, uses ones.
    T : int
        Number of periods to simulate (after burn-in).
    n_simulations : int
        Number of independent simulations.
    burn_in : int
        Initial periods to discard.
    rng : numpy.random.Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    simulated : ndarray (n_simulations, T, n)
        Simulated data. ``simulated[s, t, i]`` = value of variable i
        at period t in simulation s.
    """
    if rng is None:
        rng = np.random.default_rng()

    n = P.shape[0]
    k = Q.shape[1]

    if shocks_std is None:
        shocks_std = np.ones(k)

    total_periods = burn_in + T
    simulated = np.zeros((n_simulations, T, n))

    for s in range(n_simulations):
        x = np.zeros(n)
        sim_full = np.zeros((total_periods, n))

        for t in range(total_periods):
            # Draw structural shocks
            eps = rng.normal(0, shocks_std, size=k)

            # System evolution
            x = P @ x + Q @ eps
            sim_full[t, :] = x

        # Discard burn-in
        simulated[s, :, :] = sim_full[burn_in:, :]

    if n_simulations == 1:
        return simulated[0]

    return simulated


# =============================================================================
# Moment Computation
# =============================================================================

def compute_moments(P, Q, shocks_std=None, T=1000, rng=None):
    r"""
    Compute model-implied business cycle moments.

    Computes three types of second moments:

    1. **Standard deviations**: :math:`\sigma(x_i)`

    2. **Autocorrelations** (order 1):
       :math:`\rho_1(x_i) = \\text{Corr}(x_{i,t}, x_{i,t-1})`

    3. **Contemporaneous correlations**:
       :math:`\\rho(x_i, x_j) = \\text{Corr}(x_{i,t}, x_{j,t})`

    Parameters
    ----------
    P : ndarray (n, n)
        Transition matrix.
    Q : ndarray (n, k)
        Shock impact matrix.
    shocks_std : ndarray (k,), optional
        Shock standard deviations.
    T : int
        Simulation length.
    rng : numpy.random.Generator, optional
        Random number generator.

    Returns
    -------
    moments : dict
        Dictionary with keys:
        - ``std`` (ndarray): Standard deviations ``(n,)``
        - ``autocorr`` (ndarray): First-order autocorrelations ``(n,)``
        - ``corr`` (ndarray): Correlation matrix ``(n, n)``
        - ``data`` (ndarray): Simulated data ``(T, n)``
    """
    if rng is None:
        rng = np.random.default_rng()

    n = P.shape[0]
    k = Q.shape[1]

    if shocks_std is None:
        shocks_std = np.ones(k)

    # Simulate long sample
    data = simulate_model(P, Q, shocks_std, T=T, n_simulations=1,
                          burn_in=100, rng=rng)
    if data.ndim == 3:
        data = data[0]

    # Standard deviations
    std = np.std(data, axis=0)

    # First-order autocorrelations
    autocorr = np.zeros(n)
    for i in range(n):
        if std[i] > 1e-12:
            autocorr[i] = np.corrcoef(data[1:, i], data[:-1, i])[0, 1]
        else:
            autocorr[i] = 0.0

    # Contemporaneous correlation matrix
    corr = np.corrcoef(data.T)

    return {
        'std': std,
        'autocorr': autocorr,
        'corr': corr,
        'data': data,
    }


# =============================================================================
# Moment Matching
# =============================================================================

def moment_matching(model, params, empirical_moments, var_names=None):
    r"""
    Compare model-implied moments against empirical data moments.

    Given empirical moments :math:`\mathbf{m}_{data}` and model-implied
    moments :math:`\mathbf{m}_{model}(\theta)`, compute the distance:

    .. math::

        d(\\theta) = \|\mathbf{m}_{model}(\\theta) - \mathbf{m}_{data}\|_2

    Parameters
    ----------
    model : DSGE model instance
        Solved model.
    params : dict
        Model parameters.
    empirical_moments : dict
        Dictionary with keys matching ``compute_moments`` output:
        ``std``, ``autocorr``, ``corr``.
    var_names : list of str, optional
        Variable names to match.

    Returns
    -------
    result : dict
        Dictionary with keys:
        - ``model_moments`` (dict): Model-implied moments
        - ``distance`` (float): Euclidean distance
        - ``dist_std`` (float): Distance in standard deviations
        - ``dist_autocorr`` (float): Distance in autocorrelations
        - ``dist_corr`` (float): Distance in correlations
    """
    from dsgepy.solution import solve_model

    # Solve model
    sol = solve_model(model, params)
    P, Q = sol['P'], sol['Q']

    # Compute model moments
    model_moments = compute_moments(P, Q, T=1000)

    # Compute distances
    dist_std = np.linalg.norm(model_moments['std'] - empirical_moments['std'])
    dist_autocorr = np.linalg.norm(model_moments['autocorr'] - empirical_moments['autocorr'])
    dist_corr = np.linalg.norm(model_moments['corr'] - empirical_moments['corr'])

    distance = np.sqrt(dist_std**2 + dist_autocorr**2 + dist_corr**2)

    return {
        'model_moments': model_moments,
        'distance': distance,
        'dist_std': dist_std,
        'dist_autocorr': dist_autocorr,
        'dist_corr': dist_corr,
    }


# =============================================================================
# Kalman Filter — State Estimation
# =============================================================================

def filtered_variables(P, Q, data, shocks_std=None):
    r"""
    Estimate filtered state variables using the Kalman filter.

    The Kalman filter recursively estimates unobserved states
    :math:`\mathbf{x}_t` from a linear state-space model:

    **State Equation**:

    .. math::

        \\mathbf{x}_t = \\mathbf{P} \\mathbf{x}_{t-1} +
        \\mathbf{Q} \\boldsymbol{\\varepsilon}_t,
        \\quad \\boldsymbol{\\varepsilon}_t \\sim
        \\mathcal{N}(\\mathbf{0}, \\Sigma_\\varepsilon)

    **Observation Equation** (identity, full observability):

    .. math::

        \\mathbf{y}_t = \\mathbf{x}_t + \\boldsymbol{\\nu}_t,
        \\quad \\boldsymbol{\\nu}_t \\sim \\mathcal{N}(\\mathbf{0}, \\Sigma_\\nu)

    This is the "signal extraction" or "smoothing" problem: given noisy
    data, recover the underlying structural shocks and states.

    Parameters
    ----------
    P : ndarray (n, n)
        Transition matrix.
    Q : ndarray (n, k)
        Shock impact matrix.
    data : ndarray (T, n)
        Observed data (log-deviations from steady state).
    shocks_std : ndarray (k,), optional
        Standard deviations of shocks.

    Returns
    -------
    result : dict
        Dictionary with keys:
        - ``x_filtered`` (ndarray): Filtered states ``(T, n)``
        - ``x_predicted`` (ndarray): Predicted states ``(T, n)``
        - ``P_filtered`` (ndarray): Filtered covariances ``(T, n, n)``
        - ``log_likelihood`` (float): Log-likelihood of the data
    """
    T = data.shape[0]
    n = P.shape[0]
    k = Q.shape[1]

    if shocks_std is None:
        shocks_std = np.ones(k)

    # State noise covariance
    Sigma_eps = np.diag(shocks_std ** 2)
    Q_Sigma = Q @ Sigma_eps @ Q.T

    # Observation noise (small for numerical stability)
    obs_noise = 1e-8
    R = obs_noise * np.eye(n)

    # Initialize
    x_pred = np.zeros((T, n))
    x_filt = np.zeros((T, n))
    P_pred = np.zeros((T, n, n))
    P_filt = np.zeros((T, n, n))

    # Initial state covariance (unconditional variance)
    # vec(P_0) = (I - P kron P)^{-1} * vec(Q_Sigma)
    try:
        I_n2 = np.eye(n * n)
        P_kron = np.kron(P, P)
        P0_vec = np.linalg.solve(I_n2 - P_kron, Q_Sigma.ravel('F'))
        P0 = P0_vec.reshape(n, n, order='F')
    except np.linalg.LinAlgError:
        P0 = np.eye(n)

    x_curr = np.zeros(n)
    P_curr = P0

    log_likelihood = 0.0

    for t in range(T):
        # --- Prediction step ---
        x_pred[t] = P @ x_curr
        P_pred[t] = P @ P_curr @ P.T + Q_Sigma

        # --- Update step ---
        # Kalman gain: K = P_pred * H' * inv(H * P_pred * H' + R)
        # With H = I (identity observation matrix):
        S = P_pred[t] + R   # Innovation covariance
        try:
            K = np.linalg.solve(S.T, P_pred[t].T).T  # K = P_pred * inv(S)
        except np.linalg.LinAlgError:
            K = P_pred[t] @ np.linalg.pinv(S)

        # Innovation
        innovation = data[t] - x_pred[t]

        # Update state estimate
        x_filt[t] = x_pred[t] + K @ innovation
        P_filt[t] = (np.eye(n) - K) @ P_pred[t]

        # Log-likelihood contribution
        try:
            S_inv = np.linalg.inv(S)
            _, logdet = np.linalg.slogdet(S)
            ll_term = -0.5 * (n * np.log(2 * np.pi) + logdet +
                              innovation @ S_inv @ innovation)
            log_likelihood += ll_term
        except np.linalg.LinAlgError:
            pass

        # Advance
        x_curr = x_filt[t]
        P_curr = P_filt[t]

    return {
        'x_filtered': x_filt,
        'x_predicted': x_pred,
        'P_filtered': P_filt,
        'log_likelihood': log_likelihood,
    }
