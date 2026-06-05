import numpy as np
import control as ct

from Simulation.simulation import TFSimulator

class RSTController:
    def __init__(self, R, S, T):
        """Gerenates the transfer function simulators for the R,S and T polynomials

        Args:
            R (TransferFunction): the ct.tf() of the R polynomial
            S (TransferFunction): the ct.tf() of the S polynomial
            T (TransferFunction): the ct.ts() od the T polynomial
        """
        self.R_sim = TFSimulator(R, 0)
        self.T_sim = TFSimulator(T, 0)
        self.S_inv_sim = TFSimulator(
            ct.tf([1], S.num_list[0][0], S.dt),
            0
        )
    def setReference(self,r):
        """Sets the reference of the RST controller

        Args:
            r (_float_): the reference (often 1.0)
        """
        self.reference=r
        return
    def step(self,y):
        v = (
        self.T_sim.step(self.reference) ### from reference via T
        - self.R_sim.step(y) ### from output via R
         )

        u = self.S_inv_sim.step(v) ### everything passed down to 1/S
        
        return u