import numpy as np
import control as ct

from Simulation.simulation import TFSimulator

class RSTController:
    def __init__(self, R:ct.TransferFunction, S:ct.TransferFunction, T:ct.TransferFunction):
        """Gerenates the transfer function simulators for the R,S and T polynomials

        Args:
            R (ct.TransferFunction): the ct.tf() of the R polynomial
            S (ct.TransferFunction): the ct.tf() of the S polynomial
            T (ct.TransferFunction): the ct.ts() of the T polynomial
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