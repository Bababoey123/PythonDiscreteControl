from Metrics_Plotting import SimLog

import numpy as np
import scipy.linalg as la
from scipy.linalg import toeplitz
import control as ct
import csv


def _trim(p):
    """
    Remove leading zeros safely. 
    For example if an array is of form [0,0,1], the higher order zeros are not necessary and will be trimmed out. 
    If the array is filled with zeros it will output [0.0]
    """
    p = np.array(p, dtype=float)
    return np.trim_zeros(p, 'f') if np.any(p) else np.array([0.0])


def _conv_matrix(a, n):
    """
    Builds convolution matrix so that:
        conv(a, x) = A @ x
    """
    a = np.asarray(a, dtype=float) ## build a numpy array from the list a 
    m = len(a)

    col = np.r_[a, np.zeros(n - 1)] #concacatantes an array of [n-1] zeros to a forming a coluums vector 
    row = np.r_[a[0], np.zeros(n - 1)] #same thing but with a row 

    return toeplitz(col, row)[:m + n - 1, :n] # return a toeplitz matrix of m+n-1 rows and n collumns


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

    # Normalize the polyomials 
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

    M = np.hstack([AS, BR]) # stacking horizontally the AS and BR toeplitz matrices forming the M matrix 

    # Pad / trim A_cl to match system size
    target_len = M.shape[0] # extracing the target lengh of the solution 
    A_cl = np.r_[A_cl, np.zeros(target_len - len(A_cl))] if len(A_cl) < target_len else A_cl[:target_len]
    # if the target polynomial is too low, we need to add zeros to bump up A_cl to the needed order 
    # else if the target polynomial is too high, it is trimmed down from the lower-order coeficients  
    # Solve
    theta = la.lstsq(M, A_cl)[0]# solving the linear system [AR,BR][S; R]=A_cl
    # via the least squares method, this gives the most exact approximate solution 

    S_coeffs = theta[:nS] # extracting the first coeficients into S
    R_coeffs = theta[nS:] # and the remaining into R 

    # Verify denominator matching for sanity
    A_cl_check = np.r_[A_cl, np.zeros(M.shape[0] - len(A_cl))] if len(A_cl) < M.shape[0] else A_cl[: M.shape[0]] # building the checking polynomial with the same rules 
    den_check = np.polyadd(np.polymul(A, S_coeffs), np.polymul(B, R_coeffs)) # computes AS+BR from the calculates S and R solutions
    if not np.allclose(den_check, A_cl_check, atol=1e-8): #comparing the exact and the calculated Acl against a tolerance
        raise ValueError(
            "RST denominator synthesis failed: A*S + B*R does not equal desired A_cl."
        )

    # ------------------------------------------------------------
    # 4. Compute T correctly from:
    #       B T = B_cl
    # ------------------------------------------------------------
    nT = max(1, len(B_cl) - len(B) + 1) #computes the lengh of the T vector comes from B T = B_cl, but cannot be negative 

    # Use exact polynomial division if possible
    quotient, remainder = np.polydiv(B_cl, B) ## compute T=B\Bcl
    remainder = np.trim_zeros(np.round(remainder, decimals=12), 'f') # cleaning off numerical noise and trimming zeros
    if len(remainder) == 0 or np.allclose(remainder, 0, atol=1e-8): # if the solution is exact, the remainder is almost zero
        T_coeffs = np.trim_zeros(quotient, 'f')
        if len(T_coeffs) == 0:  ## if the quotient is zero 
            T_coeffs = np.array([0.0]) ## add a zero 
        if len(T_coeffs) < nT:
            T_coeffs = np.r_[T_coeffs, np.zeros(nT - len(T_coeffs))]
    else: ## if the tolerance is not not exact we raise an error or we cannot divide B by B_cl
        raise ValueError(
            "Desired numerator is not achievable with B * T; adjust the desired transfer function or use a different plant model."
        )

    # If the polynomial division returns higher-order coefficients, keep the full result.
    nT = len(T_coeffs)
    # ------------------------------------------------------------
    # 5. Transfer functions 
    # ------------------------------------------------------------
    S_tf = ct.tf(S_coeffs, [1], plant_discrete_tf.dt)
    R_tf = ct.tf(R_coeffs, [1], plant_discrete_tf.dt)
    T_tf = ct.tf(T_coeffs, [1], plant_discrete_tf.dt)

    # ------------------------------------------------------------
    # 6. Verification 
    # ------------------------------------------------------------

    num_cl = np.polymul(B, T_coeffs) # =B_cl
    den_cl = np.polyadd(np.polymul(A, S_coeffs), np.polymul(B, R_coeffs)) #=A_cl

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


