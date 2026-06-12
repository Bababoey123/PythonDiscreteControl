import numpy as np
import control as ct

from Simulation.simulation import TFSimulator


class RSTController:
    """Polynomial RST controller implementing the two-degree-of-freedom control law:

        S(z) · u[k] = T(z) · r[k] − R(z) · y[k]

    Equivalently:
        v[k] = T·r_hist − R·y_hist      (FIR dot products on reference and output histories)
        u[k] = (1/S) · v[k]             (recursive IIR filter realised by TFSimulator)

    The two degrees of freedom mean that T and R can be tuned independently: T shapes
    the reference tracking response while R shapes disturbance rejection without affecting
    the other.
    """

    def __init__(self, R: ct.TransferFunction, S: ct.TransferFunction, T: ct.TransferFunction):
        """Initialises the RST controller from three polynomial transfer functions.

        The polynomials are typically computed by ``Compute_Desired_RST`` or
        ``Compute_Denominator_Matching_RST`` in ``Utils/computeRST.py``.

        Args:
            R (ct.TransferFunction): Feedback polynomial — multiplies the output history y[k].
            S (ct.TransferFunction): Denominator polynomial — defines the recursive (IIR) filter
                on the intermediate signal v[k].  Must be monic.
            T (ct.TransferFunction): Feedforward polynomial — multiplies the reference history r[k].
        """
        self.R_coeffs = R.num_list[0][0]
        self.S_coeffs = S.num_list[0][0]
        self.T_coeffs = T.num_list[0][0]
        dt = S.dt

        self.r_hist = np.zeros(len(self.T_coeffs))
        self.y_hist = np.zeros(len(self.R_coeffs))

        one_over_S = ct.tf([1.0], self.S_coeffs, dt)
        self.S_block_sim = TFSimulator(one_over_S, 0)

        self.reference = 0.0

    def reset(self):
        """Resets the controller's internal state to zero.

        Clears the reference history, output history, and the 1/S recursive filter state.
        Call this before each new simulation run to avoid carry-over from a previous run.
        """
        self.r_hist[:] = 0.0
        self.y_hist[:] = 0.0
        self.S_block_sim.reset(0)

    def setReference(self, r):
        """Sets the reference setpoint for the controller.

        Args:
            r (float): Desired plant output (setpoint).
        """
        self.reference = r

    def step(self, y):
        """Computes the control output u[k] for the current plant measurement y[k].

        Implements one step of the RST control law:
            v[k] = T·r_hist − R·y_hist
            u[k] = (1/S) · v[k]

        Args:
            y (float): Current plant output y[k].

        Returns:
            float: Control signal u[k].
        """
        # Shift reference history and insert the current setpoint at index 0
        self.r_hist[1:] = self.r_hist[:-1]
        self.r_hist[0] = self.reference

        # Shift output history and insert the current measurement at index 0
        self.y_hist[1:] = self.y_hist[:-1]
        self.y_hist[0] = y

        v_k = np.dot(self.T_coeffs, self.r_hist) - np.dot(self.R_coeffs, self.y_hist)
        u_k = self.S_block_sim.step(v_k)

        return u_k
