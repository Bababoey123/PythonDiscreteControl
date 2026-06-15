import numpy as np
import control as ct

import computeRST


def trim(p):
    arr = np.array(p, dtype=float)
    return np.trim_zeros(arr, 'f') if np.any(arr) else np.array([0.0])


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _check_a0_factor(poly, A0, label):
    """Assert that A0 divides poly exactly (zero remainder)."""
    _, rem = np.polydiv(poly, A0)
    rem = np.trim_zeros(np.round(rem, 12), 'f')
    assert len(rem) == 0 or np.allclose(rem, 0, atol=1e-7), \
        f"{label} does not contain A0 as a factor (remainder: {rem})"


def _check_diophantine(A, B, S, R, A_cl_total, label):
    """Assert A*S + B*R == A_cl_total."""
    den_check = np.polyadd(np.polymul(A, S), np.polymul(B, R))
    n = max(len(den_check), len(A_cl_total))
    den_check = np.r_[np.zeros(n - len(den_check)), den_check]
    target    = np.r_[np.zeros(n - len(A_cl_total)), A_cl_total]
    assert np.allclose(den_check, target, atol=1e-7), \
        f"{label}: A*S + B*R does not equal A_m * A0 (max error {np.max(np.abs(den_check - target)):.2e})"


# ---------------------------------------------------------------------------
# Test 1 – Compute_Desired_RST with A0 (no integrator, first-order plant)
#
# Plant: deg_A = deg_B = 1  =>  target_len = 3  =>  deg(A_m) <= 2
# A_m degree 2 + A0 degree 1: total char poly degree 3.
# B_cl = 0.5 * B so that B_cl / B = 0.5 (scalar T').
# ---------------------------------------------------------------------------
def test_compute_desired_rst_with_A0():
    plant = ct.tf([0.1, 0.05], [1.0, -0.8], 0.1)
    A0    = np.array([1.0, -0.5])            # observer pole at z = 0.5
    A_m   = np.array([1.0, -1.0, 0.2])       # dominant poles (degree 2)
    B     = trim(plant.num[0][0])
    B_cl  = 0.5 * B                          # B_cl divisible by B (T' = 0.5)
    Desired_TF = ct.tf(B_cl, A_m, 0.1)      # desired TF uses A_m only (no A0)

    S_tf, R_tf, T_tf, H_cl = computeRST.Compute_Desired_RST(
        Desired_TF, plant, Integrator=False, A0=A0)

    S = trim(S_tf.num[0][0])
    R = trim(R_tf.num[0][0])
    T = trim(T_tf.num[0][0])

    # Landau property: A0 must divide S, R, and T
    _check_a0_factor(S, A0, "S")
    _check_a0_factor(R, A0, "R")
    _check_a0_factor(T, A0, "T")

    # Diophantine: A*S + B*R = A_m * A0
    A_norm    = trim(plant.den[0][0]); A_norm = A_norm / A_norm[0]
    B_plant   = trim(plant.num[0][0])
    A_cl_total = np.convolve(A_m, A0)
    _check_diophantine(A_norm, B_plant, S, R, A_cl_total, "Compute_Desired_RST with A0")

    # Closed-loop should match Desired_TF (A0 cancels from H_cl)
    assert np.isclose(ct.dcgain(H_cl), 0.5 * np.sum(B) / np.sum(A_m), atol=1e-6), \
        "DC gain mismatch"

    print("test_compute_desired_rst_with_A0 PASSED.")


# ---------------------------------------------------------------------------
# Test 2 – Compute_Denominator_Matching_RST with A0 (no integrator)
#
# Same plant/polynomial choice as test 1.
# T = t0 * A0  =>  H_ry = t0*B / A_m  with DC gain = 1.
# ---------------------------------------------------------------------------
def test_compute_denominator_matching_rst_with_A0():
    plant = ct.tf([0.1, 0.05], [1.0, -0.8], 0.1)
    A0    = np.array([1.0, -0.5])
    A_m   = np.array([1.0, -1.0, 0.2])   # dominant polynomial only, NOT A_m * A0

    S_tf, R_tf, T_tf, H_cl = computeRST.Compute_Denominator_Matching_RST(
        A_m, plant, Integrator=False, A0=A0)

    S = trim(S_tf.num[0][0])
    R = trim(R_tf.num[0][0])
    T = trim(T_tf.num[0][0])

    # A0 must divide T (cancels A0 from H_cl numerator).
    # S and R come directly from the full Diophantine — A0 need not divide them.
    _check_a0_factor(T, A0, "T")

    # Diophantine: A*S + B*R = A_m * A0
    A_norm    = trim(plant.den[0][0]); A_norm = A_norm / A_norm[0]
    B_plant   = trim(plant.num[0][0])
    A_cl_total = np.convolve(A_m, A0)
    _check_diophantine(A_norm, B_plant, S, R, A_cl_total,
                       "Compute_Denominator_Matching_RST with A0")

    # DC gain = 1  (reference tracking)
    assert np.isclose(ct.dcgain(H_cl), 1.0, atol=1e-6), \
        f"DC gain is {ct.dcgain(H_cl):.6f}, expected 1.0"

    # All closed-loop poles should lie inside the unit circle
    poles = np.roots(H_cl.den_list[0][0])
    assert np.all(np.abs(poles) < 1.0 + 1e-8), \
        f"Unstable closed-loop poles: {poles[np.abs(poles) >= 1.0]}"

    # S must have a positive leading coefficient (required for correct 1/S sign)
    assert S[0] > 0, f"S leading coefficient is negative ({S[0]:.4f}); simulation would diverge"

    print("test_compute_denominator_matching_rst_with_A0 PASSED.")


# ---------------------------------------------------------------------------
# Test 3 – Compute_Denominator_Matching_RST with integrator, no A0
#
# Baseline test: A0 = [1] (identity), integrator embedded in S.
# With A0 = identity: S = S', R = R', T = [t0]  — identical to old behavior.
# ---------------------------------------------------------------------------
def test_compute_denominator_matching_rst_with_integrator():
    plant = ct.tf([0.1, 0.05], [1.0, -0.8], 0.1)
    A_cl  = np.array([1.0, -1.2, 0.45])

    S_tf, R_tf, T_tf, H_cl = computeRST.Compute_Denominator_Matching_RST(
        A_cl, plant, Integrator=True)

    S = trim(S_tf.num[0][0])
    R = trim(R_tf.num[0][0])

    # S must contain (z-1) as a factor
    _, rem = np.polydiv(S, np.array([1.0, -1.0]))
    assert np.allclose(np.trim_zeros(rem, 'f'), 0, atol=1e-8), \
        "S does not contain the integrator factor (z-1)"

    # Diophantine: A*S + B*R = A_cl  (A0 = 1, so A_cl_total = A_cl)
    A_norm  = trim(plant.den[0][0]); A_norm = A_norm / A_norm[0]
    B_plant = trim(plant.num[0][0])
    _check_diophantine(A_norm, B_plant, S, R, A_cl,
                       "Compute_Denominator_Matching_RST with integrator")

    print("test_compute_denominator_matching_rst_with_integrator PASSED.")


# ---------------------------------------------------------------------------
# Test 4 – Compute_Denominator_Matching_RST with A0 AND integrator
#
# Plant: discrete double integrator  B(z)/A(z) with deg_A=2, deg_B=1.
# With Integrator=True: target_len=4, A_m degree 2 fits (len=3 <= 4).
# A0 degree 1: total char poly degree 3.
# ---------------------------------------------------------------------------
def test_compute_denominator_matching_rst_with_A0_and_integrator():
    # Approximate ZOH double integrator (Ts=0.1 s): B/A ~ 0.005*(z+1)/(z-1)^2
    plant = ct.tf([0.005, 0.005], [1.0, -2.0, 1.0], 0.1)
    A0    = np.array([1.0, -0.7])              # observer pole at z = 0.7
    A_m   = np.array([1.0, -1.6, 0.64])        # dominant double pole at z = 0.8

    S_tf, R_tf, T_tf, H_cl = computeRST.Compute_Denominator_Matching_RST(
        A_m, plant, Integrator=True, A0=A0)

    S = trim(S_tf.num[0][0])
    R = trim(R_tf.num[0][0])
    T = trim(T_tf.num[0][0])

    # A0 must divide T. S and R come from the full Diophantine (no A0 factorisation).
    _check_a0_factor(T, A0, "T")

    # S must contain (z-1) as a factor (integrator)
    _, rem_int = np.polydiv(S, np.array([1.0, -1.0]))
    assert np.allclose(np.trim_zeros(rem_int, 'f'), 0, atol=1e-7), \
        "S does not contain the integrator factor (z-1)"

    # Diophantine: A*S + B*R = A_m * A0
    A_norm    = trim(plant.den[0][0]); A_norm = A_norm / A_norm[0]
    B_plant   = trim(plant.num[0][0])
    A_cl_total = np.convolve(A_m, A0)
    _check_diophantine(A_norm, B_plant, S, R, A_cl_total,
                       "Compute_Denominator_Matching_RST with A0+integrator")

    # DC gain = 1
    assert np.isclose(ct.dcgain(H_cl), 1.0, atol=1e-5), \
        f"DC gain is {ct.dcgain(H_cl):.6f}, expected 1.0"

    # S must have a positive leading coefficient (required for correct 1/S sign)
    assert S[0] > 0, f"S leading coefficient is negative ({S[0]:.4f}); simulation would diverge"

    print("test_compute_denominator_matching_rst_with_A0_and_integrator PASSED.")


# ---------------------------------------------------------------------------

if __name__ == '__main__':
    test_compute_desired_rst_with_A0()
    test_compute_denominator_matching_rst_with_A0()
    test_compute_denominator_matching_rst_with_integrator()
    test_compute_denominator_matching_rst_with_A0_and_integrator()
    print("\nAll computeRST tests PASSED.")
