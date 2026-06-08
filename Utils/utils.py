from Metrics_Plotting import SimLog

import scipy.linalg as la
import control as ct 
import numpy as np
import csv
def as_csv(csv_title:str,logs):
    with open(csv_title+'.csv', 'w', newline='') as csvfile:
        fieldnames = ['time', 'output','input']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for i in range(0,len(logs.t_hist)) :
            writer.writerow({'time': logs.t_hist[i] ,'output': logs.y_hist[i],'input':logs.u_hist[i]})
    return

import control as ct
import numpy as np
import scipy.linalg as la


def Compute_Desired_RST(
    Desired_TF: ct.TransferFunction, plant_discrete_tf: ct.TransferFunction
):
    """Computes the causal discrete R, S, T polynomials to obtain the desired

    closed loop transfer function. The sampling time of the plant and the
    desired TF must match.
    """

    # Extract Plant Polynomials
    A = np.array(plant_discrete_tf.den[0][0], dtype=float)
    B = np.array(plant_discrete_tf.num[0][0], dtype=float)

    # Extract Desired Closed-Loop Polynomials
    A_cl = np.array(Desired_TF.den[0][0], dtype=float)
    B_cl = np.array(Desired_TF.num[0][0], dtype=float)
    
    # Force an integrator into the plant model BEFORE synthesis
    # This forces the solver to put an integrator in S
    #z_minus_1 = np.array([1, -1]) 
    #A = np.convolve(A, z_minus_1)

    deg_A = len(A) - 1
    deg_B = len(B) - 1
    

    #minimum-degree causal controller
    deg_S = deg_B-1
    deg_R = deg_A-1

    # Required closed-loop denominator polynomial degree
    deg_cl_required = deg_A + deg_S
    matrix_size = deg_cl_required + 1

    num_S_coeffs = deg_S + 1
    num_R_coeffs = deg_R + 1

    # --- 2. ROBUST TARGET POLYNOMIAL PADDING ---
    # If the desired A_cl degree is too low, pad with trailing zeros.
    # In descending powers of z, trailing zeros implicitly insert stable observer poles at the origin (z=0).
    if len(A_cl) < matrix_size:
        padding = matrix_size - len(A_cl)
        A_cl = np.concatenate([A_cl, np.zeros(padding)])
    elif len(A_cl) > matrix_size:
        raise ValueError(
            f"Target closed-loop degree too high! Max allowed length is {matrix_size}, got {len(A_cl)}"
        )

    # --- 3. BUILD THE GENERALIZED SYLVESTER MATRIX ---
    M_Sylvester = np.zeros((matrix_size, matrix_size))

    # Fill S columns (Left side blocks)
    for j in range(num_S_coeffs):
        M_Sylvester[j : j + len(A), j] = A

    # Fill R columns (Right side blocks, properly shifted to line up with descending powers of z)
    b_row_offset = deg_cl_required - (deg_B + deg_R)
    for j in range(num_R_coeffs):
        col_idx = num_S_coeffs + j
        row_start = b_row_offset + j
        M_Sylvester[row_start : row_start + len(B), col_idx] = B

    # --- 4. SOLVE THE DIOPHANTINE SYSTEM ---
    try:
        theta = la.solve(M_Sylvester, A_cl)
    except la.LinAlgError:
        raise ValueError(
            "Sylvester matrix is singular. Check for pole-zero cancellations in your plant."
        )

    # Extract the solutions cleanly based on our newly assigned causal dimensions
    S_coeffs = theta[:num_S_coeffs]
    R_coeffs = theta[num_S_coeffs:]
    scale = S_coeffs[0]
    # --- 5. MATHEMATICALLY EXACT TRACKING SCALE (T) ---
    # Evaluate polynomials at z = 1 (steady-state DC gain check)
    A_cl_z1 = np.sum(A_cl)
    B_z1 = np.sum(B)
    B_cl_z1 = np.sum(B_cl)

    # Exact unity-gain scale condition: T(1) = A_cl(1) / B(1)
    if abs(B_z1 * B_cl_z1) > 1e-12:
        scaling_factor = A_cl_z1 / (B_z1 * B_cl_z1)
    else:
        scaling_factor = 1.0

    T_coeffs = B_cl * scaling_factor

    # --- 6. CONVERT TO DISCRETE TRANSFER FUNCTIONS ---
    S_tf = ct.tf(S_coeffs/scale, [1], plant_discrete_tf.dt)
    R_tf = ct.tf(R_coeffs/scale, [1], plant_discrete_tf.dt)
    T_tf = ct.tf(T_coeffs, [1], plant_discrete_tf.dt)

    # Runtime verification printouts
    cl_poles = np.roots(A_cl)
    print("\n--- New Causal RST Synthesis ---")
    print("Closed-loop poles roots:", cl_poles)
    print("Poles magnitude:", np.abs(cl_poles))
    print("S coefficients:", S_coeffs)
    print("R coefficients:", R_coeffs)
    print("T coefficients:", T_coeffs)

    return S_tf, R_tf, T_tf