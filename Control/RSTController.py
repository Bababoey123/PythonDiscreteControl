import numpy as np
import control as ct

from Simulation.simulation import TFSimulator

class RSTController:
    """Polynomial RST controller implementing the two-degree-of-freedom control law:

        S(z) · u[k] = T(z) · r[k] - R(z) · y[k]

    The 1/S block is realised as a recursive TFSimulator; T and R are applied
    as direct FIR dot products on the reference and output histories.
    """
    def __init__(self, R: ct.TransferFunction, S: ct.TransferFunction, T: ct.TransferFunction):
        """Initialises the RST controller from three polynomial transfer functions.

        Args:
            R (ct.TransferFunction): Feedback polynomial — acts on the output history.
            S (ct.TransferFunction): Denominator polynomial — defines the recursive filter
                on the control signal. Must be monic and stable.
            T (ct.TransferFunction): Feedforward polynomial — acts on the reference history.
        """
       ### building the T/S and R/S fractions 
        self.R_coeffs = R.num_list[0][0]
        self.S_coeffs = S.num_list[0][0]
        self.T_coeffs = T.num_list[0][0]
        dt = S.dt
        
        #historical arrays for the two inputs (r and y)
        self.r_hist = np.zeros(len(self.T_coeffs))
        self.y_hist = np.zeros(len(self.R_coeffs))

        # FSimulator for the 1/S(z) block
        one_over_S = ct.tf([1.0], self.S_coeffs, dt)
        self.S_block_sim = TFSimulator(one_over_S, 0)

        self.reference = 0.0
    def setReference(self,r):
        """Sets the reference of the RST controller

        Args:
            r (_float_): the reference (often 1.0)
        """
        self.reference=r
        return
    def step(self, y):
        """Computes the control output for the current plant measurement.

        Implements one step of the RST law:
            v[k] = T·r_hist - R·y_hist
            u[k] = (1/S) · v[k]

        Args:
            y (float): Current plant output y[k].

        Returns:
            float: Control signal u[k].
        """
        # shift reference history
        self.r_hist[1:] = self.r_hist[:-1]
        self.r_hist[0] = self.reference

        # shift output history and include the current measured output
        self.y_hist[1:] = self.y_hist[:-1]
        self.y_hist[0] = y

        # compute control law
        v_k = np.dot(self.T_coeffs, self.r_hist) - np.dot(self.R_coeffs, self.y_hist)

        # S-filter
        u_k = self.S_block_sim.step(v_k)

        return u_k