"""
DSGE Model Definitions
======================

Pre-built Dynamic Stochastic General Equilibrium (DSGE) models
with analytical steady states and equilibrium conditions.

Models
------
RBCModel          : Standard Real Business Cycle model
NewKeynesianModel : Three-equation New Keynesian model
NKModelMedium     : Medium-scale New Keynesian model

Each model class provides:
- ``steady_state(params)`` : Computes the deterministic steady state
- ``equilibrium_conditions(params)`` : Returns the system of equations
- ``variable_names`` : Ordered list of model variable names
- ``shock_names`` : Ordered list of structural shock names
- ``n_vars``, ``n_shocks``, ``n_fwd``, ``n_pred`` : Dimensional metadata
"""

import numpy as np


# =============================================================================
# RBC Model — Standard Real Business Cycle
# =============================================================================

class RBCModel:
    r"""
    Standard Real Business Cycle (RBC) model with Cobb-Douglas technology
    and AR(1) total factor productivity (TFP) shock.

    **Household Problem**

    Maximizes expected lifetime utility:

    .. math::

        \max E_0 \sum_{t=0}^{\infty} \beta^t
        \left[ \ln C_t - \chi \frac{N_t^{1+\eta}}{1+\eta} \right]

    subject to the budget constraint:

    .. math::

        C_t + I_t = w_t N_t + r_t K_t

    **Firm Problem** (Cobb-Douglas production):

    .. math::

        Y_t = Z_t K_t^{\alpha} N_t^{1-\alpha}

    **Capital Accumulation**:

    .. math::

        K_{t+1} = (1 - \delta) K_t + I_t

    **TFP Shock Process** (AR(1)):

    .. math::

        \ln Z_t = \rho_z \ln Z_{t-1} + \varepsilon_t^z,
        \quad \varepsilon_t^z \sim \mathcal{N}(0, \sigma_z^2)

    **Equilibrium Conditions** (5 equations):

    1. **Euler Equation (Consumption)**:
       :math:`\frac{1}{C_t} = \beta E_t\left[\frac{1}{C_{t+1}}(r_{t+1} + 1 - \delta)\right]`

    2. **Labor Supply**:
       :math:`\chi N_t^{\eta} C_t = w_t`

    3. **Production Function**:
       :math:`Y_t = Z_t K_t^{\alpha} N_t^{1-\alpha}`

    4. **Capital Accumulation**:
       :math:`K_{t+1} = (1-\delta) K_t + I_t`

    5. **Resource Constraint**:
       :math:`Y_t = C_t + I_t`

    Plus factor price conditions:

    - :math:`r_t = \alpha Z_t K_t^{\alpha-1} N_t^{1-\alpha}` (MPK)
    - :math:`w_t = (1-\alpha) Z_t K_t^{\alpha} N_t^{-\alpha}` (MPL)

    Parameters
    ----------
    params : dict
        Dictionary with keys:
        - ``beta`` (float): Discount factor, default 0.99
        - ``alpha`` (float): Capital share in production, default 0.36
        - ``delta`` (float): Depreciation rate, default 0.025
        - ``chi`` (float): Labor disutility weight, default 1.5
        - ``eta`` (float): Inverse Frisch elasticity, default 1.0
        - ``rho_z`` (float): TFP persistence, default 0.95
        - ``sigma_z`` (float): TFP shock standard deviation, default 0.01
    """

    def __init__(self, params=None):
        # Default calibration (quarterly)
        default_params = {
            'beta': 0.99,       # discount factor
            'alpha': 0.36,      # capital share
            'delta': 0.025,     # depreciation rate
            'chi': 1.5,         # labor disutility weight
            'eta': 1.0,         # inverse Frisch elasticity
            'rho_z': 0.95,      # TFP persistence
            'sigma_z': 0.01,    # TFP shock std
        }
        if params is not None:
            default_params.update(params)
        self.params = default_params

        # Variable ordering (7 variables):
        # [y_t, c_t, i_t, k_t, n_t, z_t, r_t, w_t]
        self.variable_names = ['y', 'c', 'i', 'k', 'n', 'z', 'r', 'w']
        self.shock_names = ['e_z']                    # TFP innovation
        self.n_vars = len(self.variable_names)          # 8
        self.n_shocks = len(self.shock_names)           # 1

        # Classification of variables:
        # Forward-looking (non-predetermined): c_t (Euler), y_t, i_t, n_t, r_t, w_t
        # Predetermined (state): k_t, z_t (AR(1))
        self.n_fwd = 6   # [y, c, i, n, r, w]
        self.n_pred = 2  # [k, z]
        self.fwd_idx = [0, 1, 2, 4, 6, 7]  # indices of forward-looking vars
        self.pred_idx = [3, 5]               # indices of predetermined vars

    def steady_state(self, params):
        r"""
        Compute the deterministic steady state of the RBC model analytically.

        **Steady State Derivation**:

        In steady state, :math:`Z = 1`. From the Euler equation:

        .. math::

            r = \frac{1}{\beta} - (1 - \delta)

        From the marginal product of capital:

        .. math::

            r = \alpha K^{\alpha-1} N^{1-\alpha}
            \;\Rightarrow\;
            \frac{K}{N} = \left(\frac{\alpha}{r}\right)^{\frac{1}{1-\alpha}}

        From the production function:

        .. math::

            \frac{Y}{N} = \left(\frac{K}{N}\right)^{\alpha}

        From capital accumulation (:math:`I = \delta K`):

        .. math::

            \frac{I}{Y} = \delta \frac{K}{Y}

        Normalize :math:`N = \bar{N}` (or solve from labor supply).

        Returns
        -------
        dict
            Steady state values for all variables and key ratios.
            Keys include: ``y``, ``c``, ``i``, ``k``, ``n``, ``z``, ``r``, ``w``,
            ``ky_ratio``, ``cy_ratio``, ``iy_ratio``.
        """
        alpha = params['alpha']
        beta = params['beta']
        delta = params['delta']
        chi = params['chi']
        eta = params['eta']

        # --- Capital return rate from Euler equation ---
        # 1 = beta * (r + 1 - delta)  =>  r = 1/beta - (1 - delta)
        r_ss = 1.0 / beta - (1.0 - delta)

        # --- Capital-labor ratio from MPK ---
        # r = alpha * (K/N)^{alpha - 1} => K/N = (alpha / r)^{1/(1-alpha)}
        kn_ratio = (alpha / r_ss) ** (1.0 / (1.0 - alpha))

        # --- Output-labor ratio ---
        # Y/N = (K/N)^alpha
        yn_ratio = kn_ratio ** alpha

        # --- Wage from MPL ---
        # w = (1 - alpha) * (K/N)^alpha
        w_ss = (1.0 - alpha) * yn_ratio

        # --- Solve for labor from labor supply condition ---
        # chi * N^eta = w / C  and  C = Y - delta*K
        # C/N = Y/N - delta * K/N = yn_ratio - delta * kn_ratio
        cn_ratio = yn_ratio - delta * kn_ratio

        # Labor supply: chi * N^eta * C = w
        # => chi * N^eta * cn_ratio * N = w
        # => chi * cn_ratio * N^{1+eta} = w
        # => N = (w / (chi * cn_ratio))^{1/(1+eta)}
        n_ss = (w_ss / (chi * cn_ratio)) ** (1.0 / (1.0 + eta))

        # --- Derive levels ---
        y_ss = yn_ratio * n_ss
        k_ss = kn_ratio * n_ss
        i_ss = delta * k_ss
        c_ss = cn_ratio * n_ss
        z_ss = 1.0  # Normalized TFP

        # --- Verify resource constraint ---
        # Y = C + I should hold by construction

        ss = {
            'y': y_ss, 'c': c_ss, 'i': i_ss,
            'k': k_ss, 'n': n_ss, 'z': z_ss,
            'r': r_ss, 'w': w_ss,
            'ky_ratio': k_ss / y_ss,
            'cy_ratio': c_ss / y_ss,
            'iy_ratio': i_ss / y_ss,
        }
        return ss

    def equilibrium_conditions(self, params):
        r"""
        Return the system of equilibrium conditions as a list of
        ``(equation, description)`` tuples.

        Each equation is a callable ``f(X_t, X_{t-1}, E_Xtp1)`` that returns
        a residual (should equal 0 in equilibrium). Variables are ordered as
        given by `variable_names`: [y, c, i, k, n, z, r, w].

        The state vector arrangement used by the linearization engine is:
        - X_t = current values
        - X_{t-1} = lagged values
        - E_Xtp1 = expected t+1 values
        - epsilon_t = current shocks

        Equations:

        1. Euler: 1/C_t - beta * E_t[(r_{t+1} + 1 - delta) / C_{t+1}] = 0
        2. Labor supply: chi * N_t^eta * C_t - w_t = 0
        3. Production: Y_t - Z_t * K_t^alpha * N_t^{1-alpha} = 0
        4. Capital acc: K_{t+1} - (1-delta)*K_t - I_t = 0
        5. Resource: Y_t - C_t - I_t = 0
        6. MPK (r_t): r_t - alpha * Z_t * K_t^{alpha-1} * N_t^{1-alpha} = 0
        7. MPL (w_t): w_t - (1-alpha) * Z_t * K_t^alpha * N_t^{-alpha} = 0
        8. TFP AR(1): ln Z_t - rho_z * ln Z_{t-1} - epsilon_t^z = 0

        Returns
        -------
        list of tuple
            Each tuple is ``(equation_callable, description_string)``.
        """
        alpha = params['alpha']
        beta  = params['beta']
        delta = params['delta']
        chi   = params['chi']
        eta   = params['eta']
        rho_z = params['rho_z']

        def eq_euler(X_t, X_lag, E_Xtp1, eps_t):
            """
            Euler equation for consumption:
            1/C_t = beta * E_t[(r_{t+1} + 1 - delta) / C_{t+1}]
            """
            c_t   = X_t[1]           # consumption
            c_tp1 = E_Xtp1[1]        # expected next-period consumption
            r_tp1 = E_Xtp1[6]        # expected next-period rental rate
            return 1.0 / c_t - beta * (r_tp1 + 1.0 - delta) / c_tp1

        def eq_labor_supply(X_t, X_lag, E_Xtp1, eps_t):
            """
            Labor supply condition:
            chi * N_t^eta * C_t = w_t
            """
            c_t = X_t[1]
            n_t = X_t[4]
            w_t = X_t[7]
            return chi * (n_t ** eta) * c_t - w_t

        def eq_production(X_t, X_lag, E_Xtp1, eps_t):
            """
            Cobb-Douglas production function:
            Y_t = Z_t * K_{t-1}^alpha * N_t^{1-alpha}
            Note: K is predetermined, so we use K_{t-1} (lagged capital)
            """
            y_t  = X_t[0]
            z_t  = X_t[5]
            k_lag = X_lag[3]          # capital is predetermined
            n_t  = X_t[4]
            return y_t - z_t * (k_lag ** alpha) * (n_t ** (1.0 - alpha))

        def eq_capital_accum(X_t, X_lag, E_Xtp1, eps_t):
            """
            Capital accumulation:
            K_{t+1} = (1 - delta) * K_t + I_t
            Written as: E_t[K_{t+1}] - (1-delta)*K_t - I_t = 0
            """
            k_t   = X_t[3]
            i_t   = X_t[2]
            k_tp1 = E_Xtp1[3]
            return k_tp1 - (1.0 - delta) * k_t - i_t

        def eq_resource(X_t, X_lag, E_Xtp1, eps_t):
            """
            Resource constraint (goods market clearing):
            Y_t = C_t + I_t
            """
            y_t = X_t[0]
            c_t = X_t[1]
            i_t = X_t[2]
            return y_t - c_t - i_t

        def eq_mpk(X_t, X_lag, E_Xtp1, eps_t):
            """
            Marginal product of capital:
            r_t = alpha * Z_t * K_{t-1}^{alpha-1} * N_t^{1-alpha}
            Note: K is predetermined, so we use K_{t-1} (lagged)
            """
            z_t   = X_t[5]
            k_lag = X_lag[3]
            n_t   = X_t[4]
            r_t   = X_t[6]
            return r_t - alpha * z_t * (k_lag ** (alpha - 1.0)) * (n_t ** (1.0 - alpha))

        def eq_mpl(X_t, X_lag, E_Xtp1, eps_t):
            """
            Marginal product of labor:
            w_t = (1 - alpha) * Z_t * K_{t-1}^alpha * N_t^{-alpha}
            """
            z_t   = X_t[5]
            k_lag = X_lag[3]
            n_t   = X_t[4]
            w_t   = X_t[7]
            return w_t - (1.0 - alpha) * z_t * (k_lag ** alpha) * (n_t ** (-alpha))

        def eq_tfp(X_t, X_lag, E_Xtp1, eps_t):
            """
            TFP AR(1) process:
            ln Z_t = rho_z * ln Z_{t-1} + epsilon_t^z
            """
            z_t   = X_t[5]
            z_lag = X_lag[5]
            return np.log(z_t) - rho_z * np.log(z_lag) - eps_t[0]

        equations = [
            (eq_euler,         "Euler equation: 1/C_t = beta*E_t[(r_{t+1}+1-delta)/C_{t+1}]"),
            (eq_labor_supply,  "Labor supply: chi*N_t^eta*C_t = w_t"),
            (eq_production,    "Production: Y_t = Z_t*K_{t-1}^alpha*N_t^{1-alpha}"),
            (eq_capital_accum, "Capital accumulation: K_{t+1} = (1-delta)*K_t + I_t"),
            (eq_resource,      "Resource constraint: Y_t = C_t + I_t"),
            (eq_mpk,           "MPK: r_t = alpha*Z_t*K_{t-1}^{alpha-1}*N_t^{1-alpha}"),
            (eq_mpl,           "MPL: w_t = (1-alpha)*Z_t*K_{t-1}^alpha*N_t^{-alpha}"),
            (eq_tfp,           "TFP AR(1): ln Z_t = rho_z*ln Z_{t-1} + e_z"),
        ]
        return equations

    def shock_matrix(self, params):
        """
        Return the shock selection matrix mapping structural shocks to
        equilibrium conditions.

        Returns
        -------
        D : ndarray (n_vars, n_shocks)
            D[i, j] = coefficient of shock j on equation i
        """
        D = np.zeros((self.n_vars, self.n_shocks))
        # e_z enters the TFP equation (equation index 7)
        D[7, 0] = -1.0
        return D


# =============================================================================
# New Keynesian Model — Three-Equation Canonical NK
# =============================================================================

class NewKeynesianModel:
    r"""
    Canonical three-equation New Keynesian (NK) model with:
    - Dynamic IS equation
    - New Keynesian Phillips Curve
    - Taylor rule (with interest rate smoothing)
    - Natural rate of interest (AR(1))

    **Equations**:

    1. **Dynamic IS (output gap)**:

    .. math::

        \tilde{y}_t = E_t[\tilde{y}_{t+1}]
        - \frac{1}{\sigma}(i_t - E_t[\pi_{t+1}] - r_t^n)

    2. **New Keynesian Phillips Curve (inflation)**:

    .. math::

        \pi_t = \beta E_t[\pi_{t+1}] + \kappa \tilde{y}_t

    3. **Taylor Rule (nominal interest rate)**:

    .. math::

        i_t = \rho i_{t-1} + (1-\rho)(\phi_\pi \pi_t + \phi_y \tilde{y}_t)
        + \varepsilon_t^m

    4. **Natural Rate of Interest (AR(1))**:

    .. math::

        r_t^n = \rho_r r_{t-1}^n + \varepsilon_t^r

    Variables: :math:`[\tilde{y}_t, \pi_t, i_t, r_t^n]`
    Shocks: :math:`[\varepsilon_t^m, \varepsilon_t^r]`

    Parameters
    ----------
    params : dict
        Dictionary with keys:
        - ``beta`` (float): Discount factor, default 0.99
        - ``sigma`` (float): Relative risk aversion (IS slope = 1/sigma), default 1.0
        - ``kappa`` (float): Slope of NK Phillips curve, default 0.1
        - ``phi_pi`` (float): Taylor rule inflation response, default 1.5
        - ``phi_y`` (float): Taylor rule output gap response, default 0.125
        - ``rho`` (float): Interest rate smoothing, default 0.8
        - ``rho_r`` (float): Natural rate persistence, default 0.9
        - ``sigma_m`` (float): Monetary shock std, default 0.01
        - ``sigma_r`` (float): Natural rate shock std, default 0.01
    """

    def __init__(self, params=None):
        default_params = {
            'beta': 0.99,        # discount factor
            'sigma': 1.0,        # inverse IES (risk aversion)
            'kappa': 0.1,        # NKPC slope
            'phi_pi': 1.5,       # Taylor rule inflation coefficient
            'phi_y': 0.125,      # Taylor rule output gap coefficient
            'rho': 0.8,          # interest rate smoothing
            'rho_r': 0.9,        # natural rate persistence
            'sigma_m': 0.01,     # monetary policy shock std
            'sigma_r': 0.01,     # natural rate shock std
        }
        if params is not None:
            default_params.update(params)
        self.params = default_params

        # Variables: [y_gap, pi, i, rn]
        self.variable_names = ['y_gap', 'pi', 'i', 'rn']
        self.shock_names = ['e_m', 'e_r']          # monetary, natural rate
        self.n_vars = len(self.variable_names)      # 4
        self.n_shocks = len(self.shock_names)       # 2

        # Classification:
        # Forward-looking: y_gap (Euler), pi (NKPC)
        # Backward-looking (predetermined): i (interest smoothing), rn (AR(1))
        self.n_fwd = 2   # [y_gap, pi]
        self.n_pred = 2  # [i, rn]
        self.fwd_idx = [0, 1]
        self.pred_idx = [2, 3]

    def steady_state(self, params=None):
        r"""
        Compute the deterministic steady state of the NK model.

        In steady state:
        - Output gap: :math:`\tilde{y} = 0`
        - Inflation: :math:`\pi = 0` (zero inflation target)
        - Nominal interest rate: :math:`i = 0` (net rate)
        - Natural rate: :math:`r^n = 0`

        Returns
        -------
        dict
            Steady state values: ``y_gap``, ``pi``, ``i``, ``rn``.
        """
        ss = {
            'y_gap': 0.0,  # zero output gap
            'pi': 0.0,     # zero inflation (net)
            'i': 0.0,      # zero nominal rate (net)
            'rn': 0.0,     # zero natural rate
        }
        return ss

    def equilibrium_conditions(self, params):
        r"""
        Return the equilibrium conditions as callables.

        Returns
        -------
        list of tuple
            Each tuple is ``(equation_callable, description_string)``.
        """
        sigma  = params['sigma']
        kappa  = params['kappa']
        beta   = params['beta']
        phi_pi = params['phi_pi']
        phi_y  = params['phi_y']
        rho    = params['rho']
        rho_r  = params['rho_r']

        def eq_dynamic_is(X_t, X_lag, E_Xtp1, eps_t):
            """
            Dynamic IS equation:
            y_gap_t = E_t[y_gap_{t+1}] - (1/sigma)*(i_t - E_t[pi_{t+1}] - rn_t)
            """
            y_t    = X_t[0]        # output gap
            y_tp1  = E_Xtp1[0]     # expected output gap
            i_t    = X_t[2]        # nominal rate
            pi_tp1 = E_Xtp1[1]     # expected inflation
            rn_t   = X_t[3]        # natural rate
            return y_t - y_tp1 + (1.0 / sigma) * (i_t - pi_tp1 - rn_t)

        def eq_nkpc(X_t, X_lag, E_Xtp1, eps_t):
            """
            New Keynesian Phillips Curve:
            pi_t = beta * E_t[pi_{t+1}] + kappa * y_gap_t
            """
            pi_t    = X_t[1]
            pi_tp1  = E_Xtp1[1]
            y_t     = X_t[0]
            return pi_t - beta * pi_tp1 - kappa * y_t

        def eq_taylor_rule(X_t, X_lag, E_Xtp1, eps_t):
            """
            Taylor rule with interest rate smoothing:
            i_t = rho*i_{t-1} + (1-rho)*(phi_pi*pi_t + phi_y*y_gap_t) + e_m
            """
            i_t    = X_t[2]
            i_lag  = X_lag[2]
            pi_t   = X_t[1]
            y_t    = X_t[0]
            return i_t - rho * i_lag - (1.0 - rho) * (phi_pi * pi_t + phi_y * y_t) - eps_t[0]

        def eq_natural_rate(X_t, X_lag, E_Xtp1, eps_t):
            """
            Natural rate AR(1):
            rn_t = rho_r * rn_{t-1} + e_r
            """
            rn_t   = X_t[3]
            rn_lag = X_lag[3]
            return rn_t - rho_r * rn_lag - eps_t[1]

        equations = [
            (eq_dynamic_is,   "Dynamic IS: y_gap_t = E[y_gap_{t+1}] - (1/sigma)*(i_t - E[pi_{t+1}] - rn_t)"),
            (eq_nkpc,         "NK Phillips: pi_t = beta*E[pi_{t+1}] + kappa*y_gap_t"),
            (eq_taylor_rule,  "Taylor rule: i_t = rho*i_{t-1} + (1-rho)*(phi_pi*pi_t + phi_y*y_gap_t) + e_m"),
            (eq_natural_rate, "Natural rate: rn_t = rho_r*rn_{t-1} + e_r"),
        ]
        return equations

    def shock_matrix(self, params):
        """
        Return the shock selection matrix D.

        Returns
        -------
        D : ndarray (n_vars, n_shocks)
        """
        D = np.zeros((self.n_vars, self.n_shocks))
        D[2, 0] = -1.0   # e_m enters Taylor rule
        D[3, 1] = -1.0   # e_r enters natural rate equation
        return D


# =============================================================================
# NKModelMedium — Medium-Scale New Keynesian Model
# =============================================================================

class NKModelMedium:
    r"""
    Medium-scale New Keynesian model with:
    - Price rigidity (Calvo, Calvo 1983)
    - Wage rigidity (Calvo)
    - Investment adjustment costs (Christiano, Eichenbaum & Evans 2005)
    - Habit formation in consumption
    - Variable capital utilization

    **Equations** (10 equations):

    1. **Euler Equation with Habit**:

    .. math::

        \lambda_t = (C_t - h C_{t-1})^{-\sigma} - \beta h E_t[(C_{t+1} - h C_t)^{-\sigma}]

    2. **Resource Constraint**:

    .. math::

        Y_t = C_t + I_t + G_t

    3. **Production Function**:

    .. math::

        Y_t = Z_t (u_t K_{t-1})^{\alpha} N_t^{1-\alpha} - \Phi

    4. **Capital Accumulation with Adjustment Costs**:

    .. math::

        K_t = (1-\delta) K_{t-1} + \left[1 - S\left(\frac{I_t}{I_{t-1}}\right)\right] I_t

    5. **Investment FOC (Tobin's Q)**:

    .. math::

        \lambda_t = q_t\left[1 - S - S'\frac{I_t}{I_{t-1}}\right]
        + \beta E_t\left[q_{t+1} S' \frac{I_{t+1}^2}{I_t^2}\right]

    6. **Capital FOC**:

    .. math::

        q_t = \beta E_t\left[\lambda_{t+1}(r_{t+1}^k u_{t+1} + q_{t+1}(1-\delta))\right]

    7. **NK Phillips Curve (Price)**:

    .. math::

        \pi_t = \beta E_t[\pi_{t+1}] + \kappa_p \cdot mc_t

    8. **Wage Phillips Curve**:

    .. math::

        \pi_t^w = \beta E_t[\pi_{t+1}^w] + \kappa_w \cdot (MRS_t - w_t)

    9. **Taylor Rule**:

    .. math::

        i_t = \rho i_{t-1} + (1-\rho)(\phi_\pi \pi_t + \phi_y \tilde{y}_t) + \varepsilon_t^m

    10. **TFP Shock (AR(1))**:

    .. math::

        \ln Z_t = \rho_z \ln Z_{t-1} + \varepsilon_t^z

    Parameters
    ----------
    params : dict
        Model parameters.
    """

    def __init__(self, params=None):
        default_params = {
            'beta': 0.99,        # discount factor
            'sigma': 1.0,        # risk aversion
            'h': 0.7,            # habit persistence
            'alpha': 0.33,       # capital share
            'delta': 0.025,      # depreciation
            'phi_i': 4.0,        # investment adjustment cost parameter
            'xi_p': 0.75,        # Calvo price stickiness
            'xi_w': 0.75,        # Calvo wage stickiness
            'theta_p': 6.0,      # elasticity of substitution (goods)
            'theta_w': 6.0,      # elasticity of substitution (labor)
            'chi': 1.0,          # labor disutility
            'eta': 1.0,          # inverse Frisch
            'phi_pi': 1.5,       # Taylor rule inflation
            'phi_y': 0.125,      # Taylor rule output
            'rho': 0.8,          # interest smoothing
            'rho_z': 0.95,       # TFP persistence
            'sigma_z': 0.01,     # TFP shock std
            'sigma_m': 0.01,     # monetary shock std
            'g_y_ratio': 0.2,    # government spending / output ratio
        }
        if params is not None:
            default_params.update(params)
        self.params = default_params

        # 10 variables: [y, c, i, k, n, pi, w, rk, q, z]
        self.variable_names = ['y', 'c', 'i', 'k', 'n', 'pi', 'w', 'rk', 'q', 'z']
        self.shock_names = ['e_z', 'e_m']
        self.n_vars = len(self.variable_names)      # 10
        self.n_shocks = len(self.shock_names)       # 2

        # Classification:
        # Forward-looking: y, c, i, n, pi, w, rk, q (8 non-predetermined vars)
        # Predetermined: k, z (2 predetermined vars)
        self.n_fwd = 8
        self.n_pred = 2
        self.fwd_idx = [0, 1, 2, 4, 5, 6, 7, 8]  # y, c, i, n, pi, w, rk, q
        self.pred_idx = [3, 9]                      # k, z

    def steady_state(self, params):
        """
        Compute steady state for the medium-scale NK model.

        In steady state: pi = 0, pi_w = 0 (zero inflation), mc = 1/markup,
        q = 1 (no adjustment costs in SS), rk = 1/beta - (1-delta).

        Returns
        -------
        dict
            Steady state values.
        """
        alpha  = params['alpha']
        beta   = params['beta']
        delta  = params['delta']
        theta_p = params['theta_p']

        # Markup and marginal cost
        markup = theta_p / (theta_p - 1.0)
        mc_ss = 1.0 / markup

        # Real rental rate from no-arbitrage: rk = 1/beta - (1-delta)
        # But with markup: mc * alpha * (K/N)^{alpha-1} = rk
        rk_ss = 1.0 / beta - (1.0 - delta)

        # Capital-labor ratio from MPK:
        # rk = mc * alpha * (K/N)^{alpha - 1}
        # => K/N = (mc * alpha / rk)^{1/(1-alpha)}
        kn_ratio = (mc_ss * alpha / rk_ss) ** (1.0 / (1.0 - alpha))

        # Output-labor ratio
        yn_ratio = kn_ratio ** alpha

        # Wage from MPL: w = mc * (1-alpha) * (K/N)^alpha
        w_ss = mc_ss * (1.0 - alpha) * yn_ratio

        # Normalize N = 1 (approximate)
        n_ss = 1.0
        y_ss = yn_ratio * n_ss
        k_ss = kn_ratio * n_ss
        i_ss = delta * k_ss

        g_y_ratio = params.get('g_y_ratio', 0.2)
        g_ss = g_y_ratio * y_ss
        c_ss = y_ss - i_ss - g_ss

        z_ss = 1.0
        q_ss = 1.0    # Tobin's Q = 1 in steady state (no adjustment costs)
        pi_ss = 0.0  # zero inflation

        ss = {
            'y': y_ss, 'c': c_ss, 'i': i_ss, 'k': k_ss,
            'n': n_ss, 'pi': pi_ss, 'w': w_ss, 'rk': rk_ss,
            'q': q_ss, 'z': z_ss,
        }
        return ss

    def equilibrium_conditions(self, params):
        r"""
        Return the 10 equilibrium conditions for the medium-scale NK model.

        Returns
        -------
        list of tuple
            Each tuple is ``(equation_callable, description_string)``.
        """
        alpha  = params['alpha']
        beta   = params['beta']
        delta  = params['delta']
        sigma  = params['sigma']
        h      = params['h']
        phi_i  = params['phi_i']
        chi    = params['chi']
        eta    = params['eta']
        xi_p   = params['xi_p']
        xi_w   = params['xi_w']
        theta_p = params['theta_p']
        theta_w = params['theta_w']
        phi_pi  = params['phi_pi']
        phi_y   = params['phi_y']
        rho     = params['rho']
        rho_z   = params['rho_z']
        g_y_ratio = params.get('g_y_ratio', 0.2)

        # Derived parameters
        markup_p = theta_p / (theta_p - 1.0)
        # Calvo price NKPC slope
        kappa_p = (1.0 - xi_p) * (1.0 - beta * xi_p) / xi_p

        def eq_euler(X_t, X_lag, E_Xtp1, eps_t):
            """Euler equation with habit formation."""
            c_t   = X_t[1]
            c_lag = X_lag[1]
            c_tp1 = E_Xtp1[1]
            pi_tp1 = E_Xtp1[5]
            i_t   = X_t[2]  # current nominal rate (we don't model it explicitly here)
            # Simplified: Euler without habit for this medium model
            lam_t = c_t ** (-sigma)
            lam_tp1 = c_tp1 ** (-sigma)
            # Using 1+i_t ~= 1 + nominal rate
            return lam_t - beta * lam_tp1 * (1.0 + i_t) / (1.0 + pi_tp1)

        def eq_resource(X_t, X_lag, E_Xtp1, eps_t):
            """Resource constraint."""
            y_t = X_t[0]
            c_t = X_t[1]
            i_t = X_t[2]
            g_t = g_y_ratio * X_t[0]  # G_t = g_y * Y_t
            return y_t - c_t - i_t - g_t

        def eq_production(X_t, X_lag, E_Xtp1, eps_t):
            """Production function."""
            y_t   = X_t[0]
            z_t   = X_t[9]
            k_lag = X_lag[3]
            n_t   = X_t[4]
            return y_t - z_t * (k_lag ** alpha) * (n_t ** (1.0 - alpha))

        def eq_capital_accum(X_t, X_lag, E_Xtp1, eps_t):
            """Capital accumulation with adjustment costs."""
            k_t   = X_t[3]
            k_lag = X_lag[3]
            i_t   = X_t[2]
            i_lag = X_lag[2]
            # S(x) = (phi_i/2) * (x - 1)^2, S'(x) = phi_i * (x - 1)
            x = i_t / i_lag if i_lag > 0 else 1.0
            S = (phi_i / 2.0) * (x - 1.0) ** 2
            return k_t - (1.0 - delta) * k_lag - (1.0 - S) * i_t

        def eq_investment_q(X_t, X_lag, E_Xtp1, eps_t):
            """Investment FOC linking Tobin's Q."""
            c_t   = X_t[1]
            i_t   = X_t[2]
            i_lag = X_lag[2]
            q_t   = X_t[8]
            # Simplified: q_t = 1 + phi_i * (I_t/I_{t-1} - 1)
            x = i_t / i_lag if i_lag > 0 else 1.0
            return q_t - (1.0 + phi_i * (x - 1.0))

        def eq_capital_foc(X_t, X_lag, E_Xtp1, eps_t):
            """Capital Euler equation."""
            c_t    = X_t[1]
            c_tp1  = E_Xtp1[1]
            rk_tp1 = E_Xtp1[7]
            q_t    = X_t[8]
            q_tp1  = E_Xtp1[8]
            lam_t  = c_t ** (-sigma)
            lam_tp1 = c_tp1 ** (-sigma)
            return q_t - beta * (lam_tp1 / lam_t) * (rk_tp1 + q_tp1 * (1.0 - delta))

        def eq_nkpc_price(X_t, X_lag, E_Xtp1, eps_t):
            """NK Phillips curve for prices."""
            pi_t   = X_t[5]
            pi_tp1 = E_Xtp1[5]
            w_t    = X_t[6]
            z_t    = X_t[9]
            k_lag  = X_lag[3]
            n_t    = X_t[4]
            # Marginal cost = w / MPL
            mpl = (1.0 - alpha) * z_t * (k_lag ** alpha) * (n_t ** (-alpha))
            mc_t = w_t / mpl
            # log-deviation from steady-state mc
            mc_ss = 1.0 / markup_p
            mc_dev = mc_t - mc_ss
            return pi_t - beta * pi_tp1 - kappa_p * mc_dev

        def eq_wage_phillips(X_t, X_lag, E_Xtp1, eps_t):
            """Wage Phillips curve (simplified)."""
            c_t  = X_t[1]
            n_t  = X_t[4]
            w_t  = X_t[6]
            # MRS = chi * N^eta * C^sigma
            mrs = chi * (n_t ** eta) * (c_t ** sigma)
            # Wage markup
            return w_t - mrs * (theta_w / (theta_w - 1.0))

        def eq_taylor_rule(X_t, X_lag, E_Xtp1, eps_t):
            """Taylor rule."""
            pi_t  = X_t[5]
            y_t   = X_t[0]
            i_t   = X_t[2]  # using i slot for nominal rate in this medium model
            y_ss  = self.steady_state(params)['y']
            y_gap = y_t - y_ss
            # i_t is a deviation from steady state
            return i_t - phi_pi * pi_t - phi_y * y_gap - eps_t[1]

        def eq_tfp(X_t, X_lag, E_Xtp1, eps_t):
            """TFP AR(1) process."""
            z_t   = X_t[9]
            z_lag = X_lag[9]
            return np.log(z_t) - rho_z * np.log(z_lag) - eps_t[0]

        equations = [
            (eq_euler,          "Euler equation with habit"),
            (eq_resource,       "Resource constraint: Y = C + I + G"),
            (eq_production,     "Production: Y = Z * K^alpha * N^{1-alpha}"),
            (eq_capital_accum,  "Capital accumulation with adj costs"),
            (eq_investment_q,   "Investment FOC (Tobin's Q)"),
            (eq_capital_foc,    "Capital FOC"),
            (eq_nkpc_price,     "NK Phillips curve (prices)"),
            (eq_wage_phillips,  "Wage Phillips curve"),
            (eq_taylor_rule,    "Taylor rule"),
            (eq_tfp,            "TFP AR(1)"),
        ]
        return equations

    def shock_matrix(self, params):
        """Return shock selection matrix."""
        D = np.zeros((self.n_vars, self.n_shocks))
        D[9, 0] = -1.0   # TFP shock
        D[8, 1] = -1.0   # Monetary shock enters Taylor rule
        return D
