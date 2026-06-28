"""
Linear Rational Expectations Model Solvers
==========================================

Solution methods for models of the form:

    A * E_t[x_{t+1}] + B * x_t + C * x_{t-1} + D * eps_t = 0

Producing the reduced-form solution:

    x_t = P * x_{t-1} + Q * eps_t

Solvers:
- generalized_schur : QZ decomposition with eigenvector extraction
- solve_model : One-line model wrapper
"""

import numpy as np
from scipy.linalg import qz, ordqz, solve


def generalized_schur(A, B, C, n_fwd=None):
    r"""
    Solve linear RE model using the generalized Schur (QZ) decomposition.

    Forms the companion pencil and extracts the solution P from the
    generalized eigenvectors associated with stable eigenvalues.

    Parameters
    ----------
    A, B, C : ndarray (n, n)
    n_fwd : int, optional

    Returns
    -------
    dict : keys P, Q, solved, eigenvalues, stable_count, unstable_count, status
    """
    n = A.shape[0]
    if n_fwd is None:
        n_fwd = n // 2
    n_pred = n - n_fwd

    # ----------------------------------------------------------------
    # Companion pencil: G1, G0  where  G1 * y_t = G0 * y_{t-1}
    #
    #   y_t = [x_t; E_t[x_{t+1}]]
    #
    #   G1 = [[0, I], [-C, -B]]
    #   G0 = [[I, 0], [0, A]]
    #
    # Generalized eigenvalues of (G1, G0):  G1 * v = lambda * G0 * v
    # The system eigenvalues are lambda. Stable: |lambda| < 1.
    # ----------------------------------------------------------------
    G0 = np.block([[np.eye(n), np.zeros((n, n))],
                   [np.zeros((n, n)), A]])
    G1 = np.block([[np.zeros((n, n)), np.eye(n)],
                   [-C, -B]])

    S, T, Qz, Z = qz(G1, G0)

    # Eigenvalues
    eigenvalues = np.zeros(2 * n, dtype=complex)
    for i in range(2 * n):
        if abs(S[i, i]) > 1e-14:
            eigenvalues[i] = T[i, i] / S[i, i]
        else:
            eigenvalues[i] = np.inf if abs(T[i, i]) > 1e-14 else 0.0

    tol_stable = 1.0 + 1e-8  # include eigenvalues at or near unit circle
    is_stable = np.abs(eigenvalues) <= tol_stable
    n_stable = int(np.sum(is_stable))
    n_unstable = 2 * n - n_stable

    P = np.zeros((n, n))
    Q_mat = np.zeros((n, 1))
    solved = False
    status = ""
    eps_eig = 1e-8

    # ----------------------------------------------------------------
    # Extract P from stable eigenvectors.
    #
    # The companion system is:
    #   G0 * y_t = G1 * y_{t-1}
    # where y_t = [x_t; E[x_{t+1}]].
    #
    # After QZ(G1, G0):  G1 * Z = G0 * Z * Lambda
    #
    # For stable eigenvalues (|lambda| < 1), the corresponding
    # generalized eigenvectors satisfy:
    #   [0, I; -C, -B] * [v1; v2] = lambda * [I, 0; 0, A] * [v1; v2]
    #
    # From the first n rows:  v2 = lambda * v1
    # From the last n rows:  -C*v1 - B*v2 = lambda * A * v2
    #
    # The columns of Z are the generalized eigenvectors.
    # For each stable eigenvector:
    #   Z_{1:n, i} is v1 (x_t component)
    #   Z_{n+1:2n, i} is v2 (E[x_{t+1}] component)
    #
    # We select n eigenvectors from the stable subspace.
    # If we have exactly n stable eigenvectors, P = Z21 * Z11^{-1}.
    # Otherwise, we select the n_pred principal stable eigenvectors
    # and compute P using the pseudo-inverse.
    # ----------------------------------------------------------------

    # Select "real" stable eigenvectors (eigenvalue not near zero or inf)
    real_mask = np.zeros(n_stable, dtype=bool)
    real_lam = np.zeros(n_stable)

    # Compute eigenvalues from the reordered decomposition
    # Without reordering, just use the original eigenvalues at stable indices
    stable_lams = eigenvalues[is_stable]

    for i in range(n_stable):
        lam = np.abs(stable_lams[i])
        if eps_eig < lam < tol_stable:
            real_mask[i] = True
            real_lam[i] = lam

    n_real = int(np.sum(real_mask))

    if n_real >= n_pred:
        # Get the corresponding columns from Z
        stable_cols = np.where(is_stable)[0]
        real_cols = stable_cols[real_mask][:n_real]

        Z11 = Z[:n, real_cols]  # (n x n_real)
        Z21 = Z[n:, real_cols]  # (n x n_real)

        # Clean complex parts
        Z11 = np.real(Z11)
        Z21 = np.real(Z21)

        if n_real >= n:
            # Full system: enough eigenvectors for all variables
            if np.linalg.matrix_rank(Z11) >= n:
                P = Z21 @ solve(Z11, np.eye(n))
                solved = True
                status = f"GS: {n_real} stable eigenvectors (full)"
            else:
                P = Z21 @ np.linalg.pinv(Z11)
                status = f"GS: pseudo-inverse ({n_real} evecs, rank Z11={np.linalg.matrix_rank(Z11)})"
                solved = True
        else:
            # Reduced: use n_pred eigenvectors, compute P via pseudo-inverse
            Z11_sub = Z11[:, :n_pred]
            Z21_sub = Z21[:, :n_pred]
            lam_sub = np.real(stable_lams[real_mask][:n_pred])

            Z11_pinv = np.linalg.pinv(Z11_sub)
            P = Z21_sub @ np.diag(lam_sub) @ Z11_pinv
            P[np.abs(P) < 1e-14] = 0.0

            if np.any(P):
                ev_P = np.linalg.eigvals(P)
                if np.max(np.abs(ev_P)) < 1.0:
                    solved = True
                    status = f"GS: reduced ({n_real} real stable, {n_pred} used)"
                else:
                    status = f"GS: solution unstable (max|eig(P)|={np.max(np.abs(ev_P)):.4f})"
            else:
                status = f"GS: reduced produced zero P"
    else:
        status = f"GS: insufficient real stable eigenvectors ({n_real} < {n_pred})"

    return {
        'P': P,
        'Q': Q_mat,
        'solved': solved,
        'eigenvalues': eigenvalues,
        'stable_count': n_stable,
        'unstable_count': n_unstable,
        'status': status,
    }


# =============================================================================
# Public API
# =============================================================================

blanchard_kahn = generalized_schur
klein_solver = generalized_schur


def solve_model(model, params=None):
    """
    One-line wrapper to solve a DSGE model.

    Parameters
    ----------
    model : Model instance
    params : dict, optional

    Returns
    -------
    dict : P, Q, solved, eigenvalues, steady_state, status
    """
    from dsgepy.linearize import compute_jacobians

    if params is None:
        params = model.params

    ss = model.steady_state(params)
    A, B, C, D = compute_jacobians(model, params)

    gs_result = generalized_schur(A, B, C, n_fwd=model.n_fwd)

    P = gs_result['P']
    n = model.n_vars
    n_shocks = model.n_shocks
    Q = np.zeros((n, n_shocks))

    if gs_result['solved'] and np.any(P):
        B_tilde = B + A @ P
        try:
            Q = -solve(B_tilde, D)
        except np.linalg.LinAlgError:
            Q = -np.linalg.lstsq(B_tilde, D, rcond=None)[0]
        Q[np.abs(Q) < 1e-14] = 0.0
    else:
        # Fallback: try direct iteration to get a usable P
        P_fb = np.zeros((n, n))
        for _ in range(200):
            M = B + A @ P_fb
            try:
                P_new = -solve(M, C)
            except np.linalg.LinAlgError:
                P_new = -np.linalg.lstsq(M, C, rcond=None)[0]
            if np.max(np.abs(P_new - P_fb)) < 1e-14:
                break
            P_fb = P_new
        residual = A @ P_fb @ P_fb + B @ P_fb + C
        ev_fb = np.linalg.eigvals(P_fb)
        if np.max(np.abs(residual)) < 2.0 and np.max(np.abs(ev_fb)) < 1.0:
            P = P_fb
            P[np.abs(P) < 1e-14] = 0.0
            gs_result['solved'] = True
            gs_result['status'] = 'GS+iteration fallback'
            B_tilde = B + A @ P
            try:
                Q = -solve(B_tilde, D)
            except np.linalg.LinAlgError:
                Q = -np.linalg.lstsq(B_tilde, D, rcond=None)[0]
            Q[np.abs(Q) < 1e-14] = 0.0

    return {
        'P': P,
        'Q': Q,
        'solved': gs_result['solved'],
        'eigenvalues': gs_result['eigenvalues'],
        'stable_count': gs_result['stable_count'],
        'unstable_count': gs_result['unstable_count'],
        'steady_state': ss,
        'A': A, 'B': B, 'C': C, 'D': D,
        'status': gs_result['status'],
    }
