
def as_csv(csv_title:str,logs):
    with open(csv_title+'.csv', 'w', newline='') as csvfile:
        fieldnames = ['time', 'output','input']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for i in range(0,len(logs.t_hist)) :
            writer.writerow({'time': logs.t_hist[i] ,'output': logs.y_hist[i],'input':logs.u_hist[i]})
    return

def Place_real_radius(plant_tf, pole_radius=0.85, steady_gain=1.0):
    """Build a Desired TF compatible with Compute_Desired_RST with real poles at a desired radius.
    - pole_radius: 0<r<1 place all desired poles at r (slower if r closer to 1)
    - steady_gain: closed-loop steady-state gain (scales B_cl = dc_gain * B)
    """
    if not (0.0 < pole_radius < 1.0):
        raise ValueError(
            "pole_radius must be between 0 and 1 for a stable discrete desired transfer function."
        )
    # Extract plant polynomials (trim leading zeros)
    A = np.trim_zeros(np.asarray(plant_tf.den_list[0][0]), 'f')
    B = np.trim_zeros(np.asarray(plant_tf.num_list[0][0]), 'f')
    steady_gain_target=steady_gain
    deg_A = len(A) - 1
    deg_B = len(B) - 1
    desired_deg = deg_A + deg_B

    poles = [pole_radius] * desired_deg if desired_deg > 0 else [pole_radius]
    A_cl = np.poly(poles)  # closed-loop denominator

    ## scaling the closed loop numerator to have the disred steady-state gain
    sumB = float(np.sum(B))
    sumAcl = float(np.sum(A_cl))

    if abs(sumB) < 1e-12:
        # fallback: avoid division by tiny plant gain
        scale = float(steady_gain_target)
    else:
        scale = float(steady_gain_target) * (sumAcl / sumB)

    B_cl = scale * B

    return ct.tf(B_cl, A_cl, plant_tf.dt)
import numpy as np

def poles_to_denominator(poles, check_stability=True, tol=1e-2):
    """
    Build a monic denominator polynomial from discrete-time poles.

    Parameters
    ----------
    poles : array-like
        Desired poles in the z-plane.
    check_stability : bool
        If True, warn if any pole is outside the unit circle.
    tol : float
        Threshold for removing numerical imaginary parts.

    Returns
    -------
    den : np.ndarray
        Denominator coefficients in descending powers of z.
    """

    poles = np.asarray(poles, dtype=complex)

    if check_stability:
        unstable = np.abs(poles) >= 1
        if np.any(unstable):
            raise ValueError(
                f"Unstable poles detected: {poles[unstable]}"
            )

    den = np.poly(poles)

    # Remove tiny numerical imaginary parts
    den = np.real_if_close(den)

    # If still complex, clip tiny imaginary components
    if np.iscomplexobj(den):
        if np.max(np.abs(np.imag(den))) < tol:
            den = np.real(den)

    # Normalize to monic polynomial
    den = den / den[0]

    return den