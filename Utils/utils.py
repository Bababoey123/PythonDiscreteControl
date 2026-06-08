from Metrics_Plotting import SimLog

import numpy as np
import scipy.linalg as la
from scipy.linalg import toeplitz
import control as ct
import csv


def _trim(p):
    """Remove leading zeros safely."""
    p = np.array(p, dtype=float)
    return np.trim_zeros(p, 'f') if np.any(p) else np.array([0.0])


def _conv_matrix(a, n):
    """
    Builds convolution matrix so that:
        conv(a, x) = A @ x
    """
    a = np.asarray(a, dtype=float)
    m = len(a)

    col = np.r_[a, np.zeros(n - 1)]
    row = np.r_[a[0], np.zeros(n - 1)]

    return toeplitz(col, row)[:m + n - 1, :n]


def Compute_Desired_RST(Desired_TF, plant_discrete_tf):
    """
    RST synthesis compatible with:
        v = T*r - R*y
        u = (1/S) v
    """

    # ------------------------------------------------------------
    # 1. Extract polynomials
    # ------------------------------------------------------------
    A = _trim(plant_discrete_tf.den[0][0])
    B = _trim(plant_discrete_tf.num[0][0])

    A_cl = _trim(Desired_TF.den[0][0])
    B_cl = _trim(Desired_TF.num[0][0])

    # Normalize (VERY important for consistency)
    A = A / A[0]
    B = B / A[0]
    A_cl = A_cl / A_cl[0]
    B_cl = B_cl / A_cl[0]

    deg_A = len(A) - 1
    deg_B = len(B) - 1

    # ------------------------------------------------------------
    # 2. Choose controller structure orders
    # ------------------------------------------------------------
    deg_S = deg_B
    deg_R = deg_A

    nS = deg_S + 1
    nR = deg_R + 1

    # ------------------------------------------------------------
    # 3. Build Diophantine equation:
    #       A S + B R = A_cl
    # ------------------------------------------------------------

    AS = _conv_matrix(A, nS)
    BR = _conv_matrix(B, nR)

    M = np.hstack([AS, BR])

    # Pad / trim A_cl to match system size
    target_len = M.shape[0]
    A_cl = np.r_[A_cl, np.zeros(target_len - len(A_cl))] if len(A_cl) < target_len else A_cl[:target_len]

    # Solve
    theta = la.lstsq(M, A_cl)[0]

    S_coeffs = theta[:nS]
    R_coeffs = theta[nS:]

    # Verify denominator matching for sanity
    A_cl_check = np.r_[A_cl, np.zeros(M.shape[0] - len(A_cl))] if len(A_cl) < M.shape[0] else A_cl[: M.shape[0]]
    den_check = np.polyadd(np.polymul(A, S_coeffs), np.polymul(B, R_coeffs))
    if not np.allclose(den_check, A_cl_check, atol=1e-8):
        raise ValueError(
            "RST denominator synthesis failed: A*S + B*R does not equal desired A_cl."
        )

    # ------------------------------------------------------------
    # 4. Compute T correctly from:
    #       B T = B_cl
    # ------------------------------------------------------------
    nT = max(1, len(B_cl) - len(B) + 1)

    # Use exact polynomial division if possible
    quotient, remainder = np.polydiv(B_cl, B)
    remainder = np.trim_zeros(np.round(remainder, decimals=12), 'f')
    if len(remainder) == 0 or np.allclose(remainder, 0, atol=1e-8):
        T_coeffs = np.trim_zeros(quotient, 'f')
        if len(T_coeffs) == 0:
            T_coeffs = np.array([0.0])
        if len(T_coeffs) < nT:
            T_coeffs = np.r_[T_coeffs, np.zeros(nT - len(T_coeffs))]
    else:
        raise ValueError(
            "Desired numerator is not achievable with B * T; adjust the desired transfer function or use a different plant model."
        )

    # If the polynomial division returns higher-order coefficients, keep the full result.
    nT = len(T_coeffs)
    # ------------------------------------------------------------
    # 5. Transfer functions (match your simulator)
    # ------------------------------------------------------------
    S_tf = ct.tf(S_coeffs, [1], plant_discrete_tf.dt)
    R_tf = ct.tf(R_coeffs, [1], plant_discrete_tf.dt)
    T_tf = ct.tf(T_coeffs, [1], plant_discrete_tf.dt)

    # ------------------------------------------------------------
    # 6. Verification (important for debugging)
    # ------------------------------------------------------------

    num_cl = np.polymul(B, T_coeffs)
    den_cl = np.polyadd(np.polymul(A, S_coeffs), np.polymul(B, R_coeffs))

    H_cl = ct.tf(num_cl, den_cl, plant_discrete_tf.dt)

    print("\n--- RST synthesis complete ---")
    print("S:", S_coeffs)
    print("R:", R_coeffs)
    print("T:", T_coeffs)
    print("\nClosed-loop TF check:")
    print(H_cl)

    return S_tf, R_tf, T_tf

def as_csv(csv_title:str,logs):
    with open(csv_title+'.csv', 'w', newline='') as csvfile:
        fieldnames = ['time', 'output','input']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for i in range(0,len(logs.t_hist)) :
            writer.writerow({'time': logs.t_hist[i] ,'output': logs.y_hist[i],'input':logs.u_hist[i]})
    return


