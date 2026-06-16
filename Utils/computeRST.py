import numpy as np
from scipy.linalg import toeplitz
import control as ct
import warnings


def trim(p):
    """Removes leading zeros from a polynomial coefficient array.

    Leading zeros represent unnecessary high-order terms (e.g. [0, 0, 1] is just [1]).
    If the entire array is zero the function returns [0.0] rather than an empty array,
    so that downstream polynomial operations always receive a valid input.

    Args:
        p (array-like): Polynomial coefficients in descending order of power.

    Returns:
        np.ndarray: Trimmed coefficient array with no leading zeros (minimum length 1).
    """
    p = np.array(p, dtype=float)
    return np.trim_zeros(p, 'f') if np.any(p) else np.array([0.0])


def conv_matrix(a, n):
    """Builds the convolution (Toeplitz) matrix for polynomial multiplication.

    Returns a matrix M such that  M @ x == np.convolve(a, x)  for any length-n
    vector x.  This lets the Diophantine equation A·S + B·R = A_m be written as
    a linear system M·θ = A_m and solved with least squares.

    The returned matrix has shape (len(a) + n − 1, n).

    Args:
        a (array-like): Polynomial coefficient array (the fixed polynomial).
        n (int): Length of the unknown polynomial x (number of columns).

    Returns:
        np.ndarray: Toeplitz convolution matrix of shape (len(a) + n − 1, n).
    """
    a = np.asarray(a, dtype=float)
    m = len(a)

    col = np.r_[a, np.zeros(n - 1)]
    row = np.r_[a[0], np.zeros(n - 1)]

    return toeplitz(col, row)[:m + n - 1, :n]


def Compute_Denominator_Matching_RST(A_cl: list, plant_discrete_tf, Integrator=True, A0=None):
    """RST synthesis by denominator (characteristic polynomial) matching.

    Computes S, R, T by solving the Diophantine equation and setting T = t0 · A0.
    The A0 factor in T cancels with the A0 factor in the closed-loop denominator:

        H_ry = B·T / (A·S + B·R) = B·(t0·A0) / (A_m·A0) = t0·B / A_m

    so the reference-to-output dynamics are governed by A_m alone.

    Algorithm
    ---------
    1. Build the target polynomial ``A_cl_total = A_m * A0``.
    2. **Dead-beat fill** (if needed): if ``A_cl_total`` is shorter than the
       Diophantine system capacity, trailing zeros are appended (= poles at z=0).
       This avoids a forced leading-zero in the right-hand side that would flip the
       sign of S and cause the closed-loop simulation to diverge.  A ``UserWarning``
       is issued; pass an explicit ``A0`` to control pole placement.
    3. **Direct solve** (when ``A_cl_total`` fits the system): solve
       ``A_eff · S_tilde + B · R = A_cl_total`` directly.  With no leading zeros
       the sign of S_tilde[0] is positive, as required by the 1/S filter.
    4. **Two-step Landau fallback** (when ``A_cl_total`` exceeds the system): solve
       the reduced equation against ``A_m``, then set ``S = A0 · S'``,
       ``R = A0 · R'``.
    5. Set ``T = t0 · A0``, where ``t0 = A_m(1) / B(1)`` enforces unity DC gain.

    Degree constraint
    -----------------
    ``deg(A_m)`` alone must not exceed the system capacity
    ``deg(A) + deg(B)`` (``deg(A_eff) + deg(B) - 1`` with Integrator).
    ``deg(A0)`` is unrestricted: when ``A_m · A0`` overflows, the two-step
    Landau fallback handles it automatically.

    Args:
        A_cl (list): Coefficients of the *dominant* closed-loop characteristic
            polynomial A_m(z) in descending powers of z.  Must NOT include the
            observer polynomial A0 — pass only the pole locations you want to dominate
            the transient response.
        plant_discrete_tf (ct.TransferFunction): Discrete plant transfer function B(z)/A(z).
        Integrator (bool, optional): If True, forces S to contain (z-1) to guarantee
            zero steady-state error for step references and exact rejection of constant
            disturbances (S(1) = 0).  Defaults to True.
        A0 (array-like or None, optional): Observer polynomial.  Its roots are additional
            closed-loop poles placed faster than A_m; they cancel from the reference path
            via T = t0·A0.  Pass None to let the function choose poles automatically
            (a UserWarning is issued when dead-beat fill is applied).  Defaults to None.

    Returns:
        tuple:
            S_tf (ct.TransferFunction): S polynomial as a discrete TF with denominator 1.
            R_tf (ct.TransferFunction): R polynomial as a discrete TF with denominator 1.
            T_tf (ct.TransferFunction): T polynomial as a discrete TF with denominator 1.
            H_cl (ct.TransferFunction): Closed-loop transfer function B·T / (A·S + B·R).

    Raises:
        ValueError: If A_cl degree exceeds the controller structure capacity, or if
            A_m * A0 contains unstable poles.
    """
    # ------------------------------------------------------------
    # 1. Extract and normalise plant polynomials
    # ------------------------------------------------------------
    A = trim(plant_discrete_tf.den[0][0])
    B = trim(plant_discrete_tf.num[0][0])
    a_lead = A[0]
    A = A / a_lead   # make A monic
    B = B / a_lead   # scale B consistently

    # A_cl is the dominant polynomial A_m (does NOT include A0)
    A_m = trim(np.asarray(A_cl, dtype=float))
    A_m = A_m / A_m[0]   # make monic

    A0 = trim(np.asarray(A0 if A0 is not None else [1.0], dtype=float))
    A0 = A0 / A0[0]   # make monic

    # Total desired characteristic polynomial: A_m * A0
    A_cl_total = trim(np.convolve(A_m, A0))

    deg_A = len(A) - 1
    deg_B = len(B) - 1

    INT = np.array([1.0, -1.0])

    # ============================================================================
    # 2. Choose controller structure orders (determined by plant, not by A0)
    #
    # The Diophantine is solved for S', R' of the same degrees as without A0.
    # A0 is then multiplied in after the solve (Landau two-step formulation).
    #
    # NON-INTEGRATOR:  deg(S') = deg(B),  deg(R') = deg(A)
    # INTEGRATOR:      S' = (z-1)·S_tilde',  deg(S_tilde') = deg(B)-1
    # ============================================================================
    if Integrator:
        deg_S_tilde = deg_B - 1
    else:
        deg_S_tilde = deg_B
    deg_R = deg_A
    nS = max(deg_S_tilde + 1, 1)
    nR = deg_R + 1

    # ============================================================================
    # 3. Build reduced Diophantine:  A_eff · S_tilde' + B · R' = A_m
    # ============================================================================
    if Integrator:
        AS = conv_matrix(np.convolve(A, INT), nS)
    else:
        AS = conv_matrix(A, nS)

    BR = conv_matrix(B, nR)
    M = np.hstack([AS, BR])

    # Degree constraint: A_m (dominant polynomial) must fit in the Diophantine system.
    target_len = M.shape[0]
    if len(A_m) > target_len:
        raise ValueError(
            f"Dominant polynomial A_cl (degree {len(A_m) - 1}) exceeds the system capacity "
            f"(max degree {target_len - 1} for this plant/structure). Reduce A_cl degree."
        )

    # Stability check on the total characteristic polynomial A_m * A0
    poles_total = np.roots(A_cl_total)
    if np.any(np.abs(poles_total) > 1.0 + 1e-8):
        raise ValueError(
            "A_cl * A0 contains poles outside the unit circle. "
            "Ensure all roots of A_cl and A0 are inside the unit disk."
        )
    if np.any(np.isclose(np.abs(poles_total), 1.0, atol=1e-8)):
        warnings.warn(
            "A_cl * A0 contains a pole on the unit circle. "
            "Closed-loop response may be marginally stable.",
            UserWarning
        )

    # ------------------------------------------------------------
    # Dead-beat fill: if A_cl_total is shorter than target_len, append trailing
    # zeros (= multiply by z^slack, adding poles at z=0).
    #
    # WHY: padding with LEADING zeros forces the first Diophantine equation to
    # "s0 + b·r0 = 0", giving s0 < 0 and inverting the 1/S filter sign →
    # simulation diverges.  TRAILING zeros shift the polynomial left without
    # touching the leading coefficient, so s0 = 1 - b·r0 > 0 is preserved.
    #
    # This situation arises when A_m has lower degree than the system capacity
    # (e.g. double integrator + Integrator=True + A0=None: capacity degree 3,
    # A_m degree 2 → slack = 1).  The extra z=0 pole decays in one step and
    # does not meaningfully affect the transient response.
    # ------------------------------------------------------------
    slack = target_len - len(A_cl_total)
    if slack > 0:
        A_cl_total = np.r_[A_cl_total, np.zeros(slack)]
        warnings.warn(
            f"A_m * A0 (degree {len(A_cl_total) - 1 - slack}) is {slack} degree(s) below "
            f"the Diophantine system capacity (degree {target_len - 1}). "
            f"{slack} deadbeat pole(s) at z=0 added automatically. "
            "Pass an explicit A0 to control the placement of these poles.",
            UserWarning
        )

    # ------------------------------------------------------------
    # 4. Solve the Diophantine.
    #
    #    Two strategies depending on whether A_cl_total fits in the system:
    #
    #    DIRECT (preferred when A_cl_total fits):
    #      Target = A_cl_total = A_m * A0.  The leading coefficient of the target is
    #      positive (A_m and A0 are monic), so S_tilde[0] > 0 and S[0] > 0.  A positive
    #      S[0] is required for the 1/S filter in RSTController to drive the plant in the
    #      correct direction.  Solving against A_m alone (with a leading-zero pad) forces
    #      S_tilde[0] < 0, inverts the 1/S gain, and causes the simulation to diverge.
    #
    #    TWO-STEP LANDAU (fallback when A_cl_total exceeds the system):
    #      Target = A_m, then S = A0 * S', R = A0 * R'.  Required when A_m is at the
    #      system degree limit and A_cl_total would overflow it.  For most non-integrating
    #      plants A_m exactly fills target_len, ensuring S'[0] > 0.
    # ------------------------------------------------------------
    if len(A_cl_total) <= target_len:
        # Direct solve: A_cl_total fits — target the full polynomial.
        print('Direct Solve Used')
        rhs = np.r_[np.zeros(target_len - len(A_cl_total)), A_cl_total]
        theta, *_ = np.linalg.lstsq(M, rhs, rcond=None)
        S_tilde = theta[:nS]
        R_coeffs = theta[nS:]
        if Integrator:
            S_coeffs = np.convolve(S_tilde, INT)
        else:
            S_coeffs = S_tilde
    else:
        # Two-step Landau: target A_m, then apply A0 factorisation.
        print('Two step used ')
        A_m_padded = np.r_[np.zeros(target_len - len(A_m)), A_m]
        theta, *_ = np.linalg.lstsq(M, A_m_padded, rcond=None)
        S_tilde_prime = theta[:nS]
        R_prime = theta[nS:]
        if Integrator:
            S_prime = np.convolve(S_tilde_prime, INT)
        else:
            S_prime = S_tilde_prime
        S_coeffs = np.convolve(A0, S_prime)
        R_coeffs = np.convolve(A0, R_prime)

    # Verify: A*S + B*R should equal A_cl_total = A_m * A0
    den_check = np.polyadd(np.polymul(A, S_coeffs), np.polymul(B, R_coeffs))
    A_cl_v = A_cl_total.copy()
    pad = max(len(den_check), len(A_cl_v))
    den_check_p = np.r_[np.zeros(pad - len(den_check)), den_check]
    A_cl_v_p = np.r_[np.zeros(pad - len(A_cl_v)), A_cl_v]
    if not np.allclose(den_check_p, A_cl_v_p, atol=1e-6):
        warnings.warn(
            "Numerical check: A*S + B*R does not closely match A_m * A0. "
            "The plant may be poorly conditioned.",
            UserWarning
        )

    # ------------------------------------------------------------
    # 5. Compute T = t0 · A0
    #
    #    H_ry = B·T / (A·S + B·R) = B·(t0·A0) / (A_m·A0) = t0·B / A_m
    #    The A0 factor in T cancels with the A0 factor in the denominator A_cl_total,
    #    so the reference-to-output dynamics are governed by A_m alone.
    #    DC gain condition: t0 · B(1) / A_m(1) = 1  =>  t0 = A_m(1) / B(1)
    # ------------------------------------------------------------
    B1 = np.polyval(B, 1)
    Am1 = np.polyval(A_m, 1)

    if np.isclose(Am1, 0.0, atol=1e-8):
        warnings.warn(
            "A_cl has a root at z=1; unity DC gain cannot be enforced. Using T = A0.",
            UserWarning
        )
        T_coeffs = A0.copy()
    else:
        t0 = Am1 / B1
        T_coeffs = t0 * A0

    # ------------------------------------------------------------
    # 7. Transfer functions
    # ------------------------------------------------------------
    S_tf = ct.tf(S_coeffs, [1], plant_discrete_tf.dt)
    R_tf = ct.tf(R_coeffs, [1], plant_discrete_tf.dt)
    T_tf = ct.tf(T_coeffs, [1], plant_discrete_tf.dt)

    # ------------------------------------------------------------
    # 8. Verification
    # Use A_cl_total directly for den_cl: avoids spurious leading-coefficient
    # noise from algebraic cancellation of high-degree terms in A*S + B*R.
    # ------------------------------------------------------------
    num_cl = np.polymul(B, T_coeffs)
    den_cl = A_cl_total   # == A_m * A0 by construction
    H_cl = ct.tf(num_cl, den_cl, plant_discrete_tf.dt)

    print("\n--- RST synthesis complete ---")
    print("S:", S_coeffs)
    print("R:", R_coeffs)
    print("T:", T_coeffs)
    print("\nExact closed-loop transfer function enforced by R,S,T:")
    print("Numerator coefficients:", num_cl)
    print("Denominator coefficients:", den_cl)
    print(H_cl)
    if np.isclose(np.polyval(den_cl, 1.0), 0.0, atol=1e-8):
        print("Closed-loop DC gain: undefined (pole at z=1)")
    else:
        print("Closed-loop DC gain:", ct.dcgain(H_cl))
    print("Closed-loop zeros (z):",np.roots(H_cl.num_list[0][0]))
    print("Closed-loop poles (z):", np.roots(H_cl.den_list[0][0]))
    print("Pole radii:           ", np.abs(np.roots(H_cl.den_list[0][0])))
    return S_tf, R_tf, T_tf, H_cl


def Compute_Desired_RST(Desired_TF, plant_discrete_tf, Integrator=True, A0=None):
    """RST synthesis from a fully specified desired closed-loop transfer function.

    Computes S, R, T using the Landau observer polynomial formulation.  The algorithm:

      1. Solve the *reduced* Diophantine for S', R':
             A_eff(z) · S'(z) + B(z) · R'(z) = A_m(z)
         where A_m = Desired_TF.den and A_eff = A*(z-1) when Integrator=True.

      2. Apply the observer polynomial factor:
             S = A0 · S',   R = A0 · R',   T = A0 · T'   (T' = B_cl / B)
         so that:
             H_cl = B·T / (A·S + B·R) = B·(A0·T') / (A_m·A0) = B·T' / A_m
                  = B·(B_cl/B) / A_m = B_cl / A_m = Desired_TF
         The A0 poles cancel exactly from the closed-loop reference-to-output path.

    Degree constraint
    -----------------
    ``Desired_TF.den`` (the dominant polynomial A_m) must satisfy:
        deg(A_m) ≤ deg(A) + deg(B)
    A0 may have any degree; it does not consume this budget.

    Args:
        Desired_TF (ct.TransferFunction): Desired closed-loop TF B_cl(z) / A_m(z)
            *without* the observer polynomial.  The denominator specifies dominant poles;
            A0 is appended separately.  The numerator B_cl must be exactly divisible by B.
        plant_discrete_tf (ct.TransferFunction): Discrete plant transfer function B(z)/A(z).
        Integrator (bool, optional): If True, forces S to contain (z-1).  Defaults to True.
        A0 (array-like or None, optional): Observer polynomial; roots are additional
            closed-loop poles that cancel from H_cl.  Pass None for no observer poles.

    Returns:
        tuple:
            S_tf, R_tf, T_tf (ct.TransferFunction): Polynomial TFs with denominator 1.
            H_cl (ct.TransferFunction): Verified closed-loop TF ≈ Desired_TF.

    Raises:
        ValueError: If degree constraint violated, A_m * A0 has unstable poles, or
            B_cl is not exactly divisible by B.
    """
    # ------------------------------------------------------------
    # 1. Extract and normalise polynomials
    # ------------------------------------------------------------
    A = trim(plant_discrete_tf.den[0][0])
    B = trim(plant_discrete_tf.num[0][0])
    A_m_raw = trim(Desired_TF.den[0][0])
    B_cl_raw = trim(Desired_TF.num[0][0])

    # Normalise: make A monic and scale B proportionally to preserve B/A
    a_lead = A[0]
    A = A / a_lead
    B = B / a_lead

    # Normalise desired denominator and scale B_cl proportionally to preserve B_cl/A_m
    m_lead = A_m_raw[0]
    A_m = A_m_raw / m_lead
    B_cl = B_cl_raw / m_lead

    A0 = trim(np.asarray(A0 if A0 is not None else [1.0], dtype=float))
    A0 = A0 / A0[0]

    # Total desired characteristic polynomial: A_m * A0
    A_cl_total = trim(np.convolve(A_m, A0))

    deg_A = len(A) - 1
    deg_B = len(B) - 1

    INT = np.array([1.0, -1.0])

    # ============================================================================
    # 2. Controller structure orders (same logic as Compute_Denominator_Matching_RST)
    # ============================================================================
    if Integrator:
        deg_S_tilde = deg_B - 1
    else:
        deg_S_tilde = deg_B
    deg_R = deg_A
    nS = max(deg_S_tilde + 1, 1)
    nR = deg_R + 1

    # ============================================================================
    # 3. Build reduced Diophantine:  A_eff · S_tilde' + B · R' = A_m
    # ============================================================================
    if Integrator:
        AS = conv_matrix(np.convolve(A, INT), nS)
    else:
        AS = conv_matrix(A, nS)

    BR = conv_matrix(B, nR)
    M = np.hstack([AS, BR])

    target_len = M.shape[0]
    if len(A_m) > target_len:
        raise ValueError(
            f"Desired denominator A_m (degree {len(A_m) - 1}) exceeds the system capacity "
            f"(max degree {target_len - 1}). Reduce the desired TF denominator degree. "
            "A0 observer poles do not consume this budget."
        )
    A_m_padded = np.r_[np.zeros(target_len - len(A_m)), A_m]

    # Stability check on total polynomial
    poles_total = np.roots(A_cl_total)
    if np.any(np.abs(poles_total) > 1.0 + 1e-8):
        raise ValueError(
            "A_m * A0 contains poles outside the unit circle. "
            "Ensure all roots of A_m and A0 are inside the unit disk."
        )
    if np.any(np.isclose(np.abs(poles_total), 1.0, atol=1e-8)):
        warnings.warn(
            "A_m * A0 contains a pole on the unit circle. "
            "Closed-loop response may be marginally stable.",
            UserWarning
        )

    # ------------------------------------------------------------
    # 4. Solve reduced Diophantine for S_tilde', R'
    # ------------------------------------------------------------
    theta, *_ = np.linalg.lstsq(M, A_m_padded, rcond=None)
    S_tilde_prime = theta[:nS]
    R_prime = theta[nS:]

    if Integrator:
        S_prime = np.convolve(S_tilde_prime, INT)
    else:
        S_prime = S_tilde_prime

    # ------------------------------------------------------------
    # 5. Observer polynomial factorisation:  S = A0 · S',  R = A0 · R'
    # ------------------------------------------------------------
    S_coeffs = np.convolve(A0, S_prime)
    R_coeffs = np.convolve(A0, R_prime)

    # Verify Diophantine: A*S + B*R = A_cl_total
    den_check = np.polyadd(np.polymul(A, S_coeffs), np.polymul(B, R_coeffs))
    A_cl_v = A_cl_total.copy()
    pad = max(len(den_check), len(A_cl_v))
    den_check_p = np.r_[np.zeros(pad - len(den_check)), den_check]
    A_cl_v_p = np.r_[np.zeros(pad - len(A_cl_v)), A_cl_v]
    if not np.allclose(den_check_p, A_cl_v_p, atol=1e-8):
        raise ValueError(
            "RST synthesis failed: A*S + B*R does not equal A_m * A0. "
            "Check plant conditioning or degree constraints."
        )

    # ------------------------------------------------------------
    # 6. Compute T = A0 · T'  where  T' = B_cl / B
    #
    #    H_cl = B·T / (A·S + B·R) = B·(A0·T') / (A_m·A0) = B·T'/A_m = B_cl/A_m
    #    A0 poles cancel from the closed-loop reference-to-output path.
    # ------------------------------------------------------------
    quotient, remainder = np.polydiv(B_cl, B)
    remainder = np.trim_zeros(np.round(remainder, decimals=12), 'f')
    if len(remainder) > 0 and not np.allclose(remainder, 0, atol=1e-8):
        raise ValueError(
            "Desired numerator B_cl is not exactly divisible by B. "
            "Adjust the desired transfer function or use a different plant model."
        )
    T_prime = np.trim_zeros(quotient, 'f')
    if len(T_prime) == 0:
        T_prime = np.array([0.0])
    T_coeffs = np.convolve(A0, T_prime)

    # ------------------------------------------------------------
    # 7. Transfer functions
    # ------------------------------------------------------------
    S_tf = ct.tf(S_coeffs, [1], plant_discrete_tf.dt)
    R_tf = ct.tf(R_coeffs, [1], plant_discrete_tf.dt)
    T_tf = ct.tf(T_coeffs, [1], plant_discrete_tf.dt)

    # ------------------------------------------------------------
    # 8. Verification
    # Use A_cl_total directly for den_cl: avoids spurious leading-coefficient
    # noise from algebraic cancellation of high-degree terms in A*S + B*R.
    # ------------------------------------------------------------
    num_cl = np.polymul(B, T_coeffs)
    den_cl = A_cl_total   # == A_m * A0 by construction
    H_cl = ct.tf(num_cl, den_cl, plant_discrete_tf.dt)

    print("\n--- RST synthesis complete ---")
    print("S:", S_coeffs)
    print("R:", R_coeffs)
    print("T:", T_coeffs)
    print("\nClosed-loop TF check:")
    print(H_cl)
    print("DC gain:", ct.dcgain(H_cl))
    print("Closed-loop poles (z):", np.roots(H_cl.den_list[0][0]))
    print("Pole radii:           ", np.abs(np.roots(H_cl.den_list[0][0])))
    return S_tf, R_tf, T_tf, H_cl
