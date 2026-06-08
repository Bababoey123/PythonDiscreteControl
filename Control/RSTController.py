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

        #FSimulator for the 1/S(z) block
        # This keeps a single, unified history for the control signal u
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
    def step(self,y):
        """Gereates the control signal u based on the output y and the reference

        Args:
            y (_float_): output signal

        Returns:
           u(_float_): control signal 
        """
        # 1. Shift and update reference input history safely
        if len(self.r_hist) > 1:
            self.r_hist[1:] = np.copy(self.r_hist[:-1])
        if len(self.r_hist) > 0:
            self.r_hist[0] = float(self.reference)

        # 2. Shift and update plant output feedback history safely
        if len(self.y_hist) > 1:
            self.y_hist[1:] = np.copy(self.y_hist[:-1])
        if len(self.y_hist) > 0:
            self.y_hist[0] = float(y)

        # 3. Compute the intermediate tracking/feedback mixture: v[k] = T*r - R*y
        v_k = np.dot(self.T_coeffs, self.r_hist) - np.dot(
            self.R_coeffs, self.y_hist
        )
    
        # 4. Reuse your untouched original TFSimulator.step() to compute u[k]
        # This handles the internal 1/S behavior and properly tracks u_hist
        u_k = self.S_block_sim.step(v_k)
        feedforward = np.dot(self.T_coeffs, self.r_hist)
        feedback = np.dot(self.R_coeffs, self.y_hist)
        print(f"FF: {feedforward:.4f} | FB: {feedback:.4f} | u_before_S: {v_k:.4f}")
 
        return u_k