import numpy as np
import control as ct
import computeRST


def trim(p):
    arr = np.array(p, dtype=float)
    return np.trim_zeros(arr, 'f') if np.any(arr) else np.array([0.0])


def test_compute_desired_rst_with_A0():
    plant = ct.tf([0.1, 0.05], [1.0, -0.8], 0.1)
    A0 = np.array([1.0, -0.5])
    desired_reduced = np.array([1.0, -1.0, 0.2])
    A_cl = np.convolve(A0, desired_reduced)
    B_cl = np.convolve([0.1, 0.05], [0.5])
    Desired_TF = ct.tf(B_cl, A_cl, 0.1)

    S_tf, R_tf, T_tf, H_cl = computeRST.Compute_Desired_RST(Desired_TF, plant, Integrator=False, A0=A0)

    S = trim(S_tf.num[0][0])
    R = trim(R_tf.num[0][0])

    qS, rS = np.polydiv(S, A0)
    qR, rR = np.polydiv(R, A0)
    assert np.allclose(np.trim_zeros(rS, 'f'), 0, atol=1e-8), "S does not contain A0 factor"
    assert np.allclose(np.trim_zeros(rR, 'f'), 0, atol=1e-8), "R does not contain A0 factor"

    A = trim(plant.den[0][0])
    B = trim(plant.num[0][0])
    A = A / A[0]
    den_check = np.polyadd(np.polymul(A, S), np.polymul(B, R))
    assert np.allclose(den_check, A_cl, atol=1e-8), "Closed-loop denominator mismatch for Compute_Desired_RST"

    print('Compute_Desired_RST with A0 passed.')


def test_compute_denominator_matching_rst_with_A0():
    plant = ct.tf([0.1, 0.05], [1.0, -0.8], 0.1)
    A0 = np.array([1.0, -0.5])
    desired_reduced = np.array([1.0, -1.0, 0.2])
    A_cl = np.convolve(A0, desired_reduced)

    S_tf, R_tf, T_tf, H_cl = computeRST.Compute_Denominator_Matching_RST(A_cl, plant, Integrator=False, A0=A0)

    S = trim(S_tf.num[0][0])
    R = trim(R_tf.num[0][0])

    qS, rS = np.polydiv(S, A0)
    qR, rR = np.polydiv(R, A0)
    assert np.allclose(np.trim_zeros(rS, 'f'), 0, atol=1e-8), "S does not contain A0 factor"
    assert np.allclose(np.trim_zeros(rR, 'f'), 0, atol=1e-8), "R does not contain A0 factor"

    A = trim(plant.den[0][0])
    B = trim(plant.num[0][0])
    A = A / A[0]
    den_check = np.polyadd(np.polymul(A, S), np.polymul(B, R))
    assert np.allclose(den_check, A_cl, atol=1e-8), "Closed-loop denominator mismatch for Compute_Denominator_Matching_RST"

    print('Compute_Denominator_Matching_RST with A0 passed.')


def test_compute_denominator_matching_rst_with_integrator():
    plant = ct.tf([0.1, 0.05], [1.0, -0.8], 0.1)
    A_cl = np.array([1.0, -1.2, 0.45])

    S_tf, R_tf, T_tf, H_cl = computeRST.Compute_Denominator_Matching_RST(A_cl, plant, Integrator=True)

    S = trim(S_tf.num[0][0])
    R = trim(R_tf.num[0][0])

    qS, rS = np.polydiv(S, np.array([1.0, -1.0]))
    assert np.allclose(np.trim_zeros(rS, 'f'), 0, atol=1e-8), "S does not contain the integrator factor (z-1)"

    A = trim(plant.den[0][0])
    B = trim(plant.num[0][0])
    A = A / A[0]
    den_check = np.polyadd(np.polymul(A, S), np.polymul(B, R))
    assert np.allclose(den_check, A_cl, atol=1e-8), "Closed-loop denominator mismatch for integrator Compute_Denominator_Matching_RST"

    print('Compute_Denominator_Matching_RST with integrator passed.')


if __name__ == '__main__':
    test_compute_desired_rst_with_A0()
    test_compute_denominator_matching_rst_with_A0()
    test_compute_denominator_matching_rst_with_integrator()
    print('All computeRST A0 tests passed.')
