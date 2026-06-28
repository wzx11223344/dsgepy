"""
Bayesian MCMC Estimation (Lite)
===============================

Estimate DSGE model parameters using Bayesian methods with a
Random Walk Metropolis-Hastings (RWMH) algorithm.

**Bayesian Framework**

Given data :math:`\\mathbf{Y} = \\{\\mathbf{y}_t\\}_{t=1}^T` and
prior distribution :math:`p(\\boldsymbol{\\theta})`, the posterior is:

.. math::

    p(\\boldsymbol{\\theta} | \\mathbf{Y}) \\propto
    \\mathcal{L}(\\mathbf{Y} | \\boldsymbol{\\theta}) \\cdot
    p(\\boldsymbol{\\theta})

where :math:`\\mathcal{L}` is the likelihood computed via the
Kalman filter.

**Random Walk Metropolis-Hastings**

1. Initialize :math:`\\boldsymbol{\\theta}^{(0)}` at the prior mean
2. For :math:`i = 1, \\ldots, N`:
   a. Propose :math:`\\boldsymbol{\\theta}^*` from
      :math:`\\mathcal{N}(\\boldsymbol{\\theta}^{(i-1)}, c^2 \\Sigma)`
   b. Compute acceptance ratio:
      :math:`\\alpha = \\min\\left(1, \\frac{p(\\boldsymbol{\\theta}^*|\\mathbf{Y})}{p(\\boldsymbol{\\theta}^{(i-1)}|\\mathbf{Y})}\\right)`
   c. Accept with probability :math:`\\alpha`, else retain current draw

**Prior Distributions**

- `BetaPrior(a, b)` : Beta distribution (for persistence parameters)
- `GammaPrior(shape, scale)` : Gamma distribution (for shock variances)
- `NormalPrior(mu, sigma)` : Normal distribution (for structural params)
- `InvGammaPrior(shape, scale)` : Inverse Gamma (for variances)

Functions
---------
bayesian_estimation          : Full RWMH estimation
compute_marginal_likelihood  : Marginal likelihood via Kalman filter
log_posterior                : Log-posterior evaluation
"""

import numpy as np


# =============================================================================
# Prior Distribution Classes
# =============================================================================

class BetaPrior:
    r"""
    Beta distribution prior.

    .. math::

        p(\\theta) = \\frac{\\theta^{a-1} (1-\\theta)^{b-1}}{B(a, b)}

    Suitable for parameters in (0, 1), e.g., persistence parameters
    :math:`\\rho`, Calvo probabilities :math:`\\xi_p`, discount factor
    :math:`\\beta`.

    Parameters
    ----------
    a : float
        Shape parameter (alpha).
    b : float
        Shape parameter (beta).
    """

    def __init__(self, a, b):
        self.a = a
        self.b = b

    def log_pdf(self, x):
        """Log probability density at x."""
        if x <= 0 or x >= 1:
            return -np.inf
        # Using scipy would be cleaner, but we implement manually
        from scipy.special import betaln
        return (self.a - 1) * np.log(x) + (self.b - 1) * np.log(1 - x) - betaln(self.a, self.b)

    def sample(self, size=None, rng=None):
        """Draw random samples."""
        if rng is None:
            rng = np.random.default_rng()
        return rng.beta(self.a, self.b, size=size)


class GammaPrior:
    r"""
    Gamma distribution prior.

    .. math::

        p(\\theta) = \\frac{\\beta^\\alpha}{\\Gamma(\\alpha)}
        \\theta^{\\alpha-1} e^{-\\beta \\theta}

    Suitable for positive parameters, e.g., shock standard deviations,
    inverse Frisch elasticity.

    Parameters
    ----------
    shape : float
        Shape parameter (alpha, also written as k).
    scale : float
        Scale parameter (beta, also written as 1/theta).
        Note: scipy convention: scale = 1/rate.
    """

    def __init__(self, shape, scale):
        self.shape = shape   # alpha
        self.scale = scale   # beta

    def log_pdf(self, x):
        """Log probability density at x."""
        if x <= 0:
            return -np.inf
        from scipy.special import gammaln
        return (self.shape - 1) * np.log(x) - x / self.scale \
               - self.shape * np.log(self.scale) - gammaln(self.shape)

    def sample(self, size=None, rng=None):
        """Draw random samples."""
        if rng is None:
            rng = np.random.default_rng()
        return rng.gamma(self.shape, self.scale, size=size)


class NormalPrior:
    r"""
    Normal (Gaussian) distribution prior.

    .. math::

        p(\\theta) = \\frac{1}{\\sigma \\sqrt{2\\pi}}
        \\exp\\left(-\\frac{(\\theta - \\mu)^2}{2\\sigma^2}\\right)

    Suitable for parameters that can take any real value, e.g.,
    Taylor rule coefficients.

    Parameters
    ----------
    mu : float
        Mean.
    sigma : float
        Standard deviation.
    """

    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def log_pdf(self, x):
        """Log probability density at x."""
        return -0.5 * ((x - self.mu) / self.sigma) ** 2 \
               - np.log(self.sigma) - 0.5 * np.log(2 * np.pi)

    def sample(self, size=None, rng=None):
        """Draw random samples."""
        if rng is None:
            rng = np.random.default_rng()
        return rng.normal(self.mu, self.sigma, size=size)


class InvGammaPrior:
    r"""
    Inverse Gamma distribution prior.

    .. math::

        p(\\theta) = \\frac{\\beta^\\alpha}{\\Gamma(\\alpha)}
        \\theta^{-\\alpha-1} e^{-\\beta / \\theta}

    Suitable for variance parameters.

    Parameters
    ----------
    shape : float
        Shape parameter (alpha).
    scale : float
        Scale parameter (beta).
    """

    def __init__(self, shape, scale):
        self.shape = shape
        self.scale = scale

    def log_pdf(self, x):
        """Log probability density at x."""
        if x <= 0:
            return -np.inf
        from scipy.special import gammaln
        return -(self.shape + 1) * np.log(x) - self.scale / x \
               + self.shape * np.log(self.scale) - gammaln(self.shape)


# =============================================================================
# Marginal Likelihood via Kalman Filter
# =============================================================================

def compute_marginal_likelihood(P, Q, shocks_std, data):
    r"""
    Compute marginal (integrated) likelihood of the data using the
    Kalman filter.

    For a linear Gaussian state-space model:

    .. math::

        \\mathbf{x}_t = \\mathbf{P} \\mathbf{x}_{t-1} +
        \\mathbf{Q} \\boldsymbol{\\varepsilon}_t

        \\mathbf{y}_t = \\mathbf{H} \\mathbf{x}_t +
        \\boldsymbol{\\nu}_t

    with :math:`\\boldsymbol{\\varepsilon}_t \\sim \\mathcal{N}(0, \\Sigma_\\varepsilon)`
    and :math:`\\boldsymbol{\\nu}_t \\sim \\mathcal{N}(0, \\Sigma_\\nu)`.

    The log-likelihood is:

    .. math::

        \\log \\mathcal{L} = -\\frac{NT}{2}\\log(2\\pi) -
        \\frac{1}{2} \\sum_{t=1}^T \\left[\\log|S_t| +
        \\mathbf{v}_t' S_t^{-1} \\mathbf{v}_t\\right]

    where :math:`\\mathbf{v}_t` is the innovation and
    :math:`S_t` its covariance at time t.

    Parameters
    ----------
    P : ndarray (n, n)
        Transition matrix.
    Q : ndarray (n, k)
        Shock impact matrix.
    shocks_std : ndarray (k,)
        Shock standard deviations.
    data : ndarray (T, n)
        Observed data (log-deviations from steady state).

    Returns
    -------
    log_likelihood : float
        Log marginal likelihood.
    """
    T = data.shape[0]
    n = P.shape[0]

    # State noise covariance
    Sigma_eps = np.diag(shocks_std ** 2)
    Q_Sigma = Q @ Sigma_eps @ Q.T

    # Observation noise (very small — near-exact measurement)
    R = 1e-8 * np.eye(n)

    # Initialize
    x = np.zeros(n)
    # Unconditional state covariance
    try:
        I_n2 = np.eye(n * n)
        P_kron = np.kron(P, P)
        P_cov = np.linalg.solve(I_n2 - P_kron, Q_Sigma.ravel('F')).reshape(n, n, order='F')
    except np.linalg.LinAlgError:
        P_cov = np.eye(n)

    log_lik = 0.0

    for t in range(T):
        # Prediction
        x_pred = P @ x
        P_pred = P @ P_cov @ P.T + Q_Sigma

        # Innovation covariance
        S = P_pred + R

        # Innovation
        v = data[t] - x_pred

        try:
            # Log determinant of S
            _, logdet_S = np.linalg.slogdet(S)

            # Solve for S^{-1} * v
            S_inv_v = np.linalg.solve(S, v)

            log_lik += -0.5 * (n * np.log(2 * np.pi) + logdet_S + v @ S_inv_v)
        except np.linalg.LinAlgError:
            # Fallback: use pseudo-inverse
            S_pinv = np.linalg.pinv(S)
            _, logdet_S = np.linalg.slogdet(S + 1e-8 * np.eye(n))
            log_lik += -0.5 * (n * np.log(2 * np.pi) + logdet_S + v @ S_pinv @ v)

        # Update
        # Kalman gain
        try:
            K = np.linalg.solve(S, P_pred.T).T
        except np.linalg.LinAlgError:
            K = P_pred @ np.linalg.pinv(S)

        x = x_pred + K @ v
        P_cov = (np.eye(n) - K) @ P_pred

    return log_lik


# =============================================================================
# Log Posterior
# =============================================================================

def log_posterior(params_vector, param_names, model, data, priors):
    r"""
    Evaluate the log-posterior at a given parameter vector.

    .. math::

        \\log p(\\boldsymbol{\\theta} | \\mathbf{Y}) =
        \\log \\mathcal{L}(\\mathbf{Y} | \\boldsymbol{\\theta}) +
        \\sum_i \\log p(\\theta_i) + \\text{const}

    Parameters
    ----------
    params_vector : ndarray
        Parameter values to evaluate.
    param_names : list of str
        Names of parameters (in order of params_vector).
    model : DSGE model instance
    data : ndarray (T, n)
        Observed data.
    priors : dict
        Mapping from parameter name to Prior object.

    Returns
    -------
    lp : float
        Log-posterior value. Returns -inf if the solution does not exist.
    """
    from dsgepy.solution import solve_model

    # Build parameter dict
    params = {}
    for i, name in enumerate(param_names):
        params[name] = params_vector[i]

    # Log-prior
    log_prior = 0.0
    for name in param_names:
        if name in priors:
            lp = priors[name].log_pdf(params[name])
            if np.isinf(lp):
                return -np.inf
            log_prior += lp

    # Solve model and compute likelihood
    try:
        sol = solve_model(model, params)
        if not sol['solved']:
            return -np.inf

        P, Q = sol['P'], sol['Q']
        shocks_std = _get_shocks_std(model, params)
        log_lik = compute_marginal_likelihood(P, Q, shocks_std, data)
    except Exception:
        return -np.inf

    if np.isnan(log_lik) or np.isinf(log_lik):
        return -np.inf

    return log_lik + log_prior


def _get_shocks_std(model, params):
    """Extract shock standard deviations from model parameters."""
    # Try common shock std parameter names
    std_names = ['sigma_z', 'sigma_m', 'sigma_r', 'sigma_g',
                 'sigma_a', 'sigma_b', 'sigma', 'sigma_eps']
    stds = []
    for name in std_names:
        if name in params:
            stds.append(params[name])

    if len(stds) == 0:
        stds = [0.01] * model.n_shocks

    return np.array(stds[:model.n_shocks])


# =============================================================================
# Random Walk Metropolis-Hastings
# =============================================================================

def bayesian_estimation(model, data, prior_dict, n_iter=5000,
                        burn_in=1000, scale=None, rng=None):
    r"""
    Bayesian estimation using Random Walk Metropolis-Hastings (RWMH).

    Algorithm:

    .. math::

        \\boldsymbol{\\theta}^{(i)} \\sim
        \\mathcal{N}(\\boldsymbol{\\theta}^{(i-1)}, c^2 \\Sigma)

    where :math:`c` is a scaling parameter adjusted to achieve a
    target acceptance rate of 20-30%, and :math:`\\Sigma` is the
    proposal covariance matrix.

    Parameters
    ----------
    model : DSGE model instance
        Model to estimate.
    data : ndarray (T, n)
        Observed data (log-deviations from steady state).
    prior_dict : dict
        Mapping from parameter name to Prior object.
        Example:
        ``{'rho_z': BetaPrior(19, 1), 'sigma_z': InvGammaPrior(0.1, 0.1)}``
    n_iter : int
        Number of MCMC iterations (default 5000).
    burn_in : int
        Number of burn-in iterations to discard.
    scale : float, optional
        Proposal scaling factor. If None, set adaptively.
    rng : numpy.random.Generator, optional
        Random number generator.

    Returns
    -------
    result : dict
        Dictionary with keys:
        - ``chain`` (ndarray): MCMC chain ``(n_effective, n_params)``
        - ``param_names`` (list): Parameter names
        - ``acceptance_rate`` (float): Acceptance rate
        - ``log_posterior`` (ndarray): Log-posterior values
        - ``posterior_mean`` (ndarray): Posterior means
        - ``posterior_std`` (ndarray): Posterior standard deviations
        - ``credible_intervals`` (ndarray): 90% credible intervals
          ``(n_params, 2)`` = [5%, 95%]
    """
    if rng is None:
        rng = np.random.default_rng()

    param_names = list(prior_dict.keys())
    n_params = len(param_names)

    # Initialize at prior mean
    theta_curr = np.zeros(n_params)
    for i, name in enumerate(param_names):
        theta_curr[i] = _prior_mean(prior_dict[name])

    # Default scale
    if scale is None:
        scale = 0.2 / np.sqrt(n_params)  # rule of thumb

    # Proposal covariance (identity, will adapt)
    proposal_cov = (scale ** 2) * np.eye(n_params)

    # Storage
    chain = np.zeros((n_iter, n_params))
    log_post_chain = np.zeros(n_iter)
    n_accept = 0

    # Evaluate initial posterior
    lp_curr = log_posterior(theta_curr, param_names, model, data, prior_dict)

    for i in range(n_iter):
        # Propose new parameter vector
        theta_prop = theta_curr + rng.multivariate_normal(
            np.zeros(n_params), proposal_cov
        )

        # Check parameter bounds (support of priors)
        valid = True
        for j, name in enumerate(param_names):
            if isinstance(prior_dict[name], BetaPrior):
                if theta_prop[j] <= 0 or theta_prop[j] >= 1:
                    valid = False
                    break
            elif isinstance(prior_dict[name], (GammaPrior, InvGammaPrior)):
                if theta_prop[j] <= 0:
                    valid = False
                    break

        if not valid:
            # Automatically reject (outside prior support)
            chain[i, :] = theta_curr
            log_post_chain[i] = lp_curr
            continue

        # Evaluate log-posterior at proposal
        lp_prop = log_posterior(theta_prop, param_names, model, data, prior_dict)

        # Metropolis-Hastings acceptance
        if np.isinf(lp_prop):
            chain[i, :] = theta_curr
            log_post_chain[i] = lp_curr
            continue

        # Log acceptance ratio
        log_alpha = lp_prop - lp_curr

        # Accept or reject
        if np.log(rng.uniform()) < log_alpha:
            theta_curr = theta_prop
            lp_curr = lp_prop
            n_accept += 1

        chain[i, :] = theta_curr
        log_post_chain[i] = lp_curr

        # Adaptive proposal scaling (optional, update every 100 iter)
        if (i + 1) % 100 == 0 and i < burn_in:
            acc_rate = n_accept / (i + 1)
            # Adjust scale toward target acceptance of 0.25
            if acc_rate < 0.15:
                scale *= 0.9
            elif acc_rate > 0.35:
                scale *= 1.1
            proposal_cov = (scale ** 2) * np.eye(n_params)

    # Post-burn-in chain
    chain_post = chain[burn_in:, :]
    log_post_post = log_post_chain[burn_in:]

    # Posterior statistics
    posterior_mean = np.mean(chain_post, axis=0)
    posterior_std = np.std(chain_post, axis=0)

    # 90% credible intervals
    ci_lower = np.percentile(chain_post, 5, axis=0)
    ci_upper = np.percentile(chain_post, 95, axis=0)
    credible_intervals = np.column_stack([ci_lower, ci_upper])

    acceptance_rate = n_accept / n_iter

    return {
        'chain': chain_post,
        'param_names': param_names,
        'acceptance_rate': acceptance_rate,
        'log_posterior': log_post_post,
        'posterior_mean': posterior_mean,
        'posterior_std': posterior_std,
        'credible_intervals': credible_intervals,
    }


def _prior_mean(prior):
    """Compute the mean of a prior distribution."""
    if isinstance(prior, BetaPrior):
        return prior.a / (prior.a + prior.b)
    elif isinstance(prior, GammaPrior):
        return prior.shape * prior.scale
    elif isinstance(prior, NormalPrior):
        return prior.mu
    elif isinstance(prior, InvGammaPrior):
        # InvGamma mean = beta / (alpha - 1) for alpha > 1
        if prior.shape > 1:
            return prior.scale / (prior.shape - 1)
        else:
            return prior.scale  # rough approximation
    return 0.5
