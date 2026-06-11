import numpy as np
import control as ct

from Simulation.simulation import TFSimulator

class DiscretePID:
    """Discrete PID controller discretised with backward Euler.

    Supports filtered and unfiltered derivative modes. Does not implement output
    saturation. Also exposes the equivalent RST polynomial representation.
    """
    def __init__(self,kp,ki,kd,dt,text_option:str="NotFiltered"):
        """Creates an instance of the PID controller.

        Args:
            kp (float): Proportional gain.
            ki (float): Integral gain.
            kd (float): Derivative gain.
            dt (float): Sampling period in seconds, taken from the plant config file.
            text_option (str, optional): ``"filtered"`` enables a first-order derivative
                filter with N=50; any other value uses the unfiltered form.
                Defaults to ``"NotFiltered"``.
        """
        self.kp=kp
        self.ki=ki
        self.kd=kd
        self.dt=dt

        ### classical PID ###
        self.transferFunction=self.As_TransferFunction(text_option)
        self.controller_sim=TFSimulator(self.transferFunction,0)
        self.PID_TF_cont=ct.tf([kd,kp,ki],[1,0])
        
        ### As RST ###
        self.As_RST()
    
        return
    def setReference(self,r):
        """Sets the reference of the PID controller

        Args:
            r (_float_): the reference (often 1.0)
        """
        self.reference=r
        return
    def step(self, y):
        """Computes the control output for the current measurement.

        Args:
            y (float): Current plant output (measured value).

        Returns:
            float: Control signal u[k].
        """
        e=self.reference-y
        return self.controller_sim.step(e)
    
    def As_TransferFunction(self, text_option: str):
        """Computes the coefficients of the discrete PID transfer function.

        Args:
            text_option (str): ``"filtered"`` builds the TF as P + I + D with a
                first-order derivative filter (N=50); any other value uses the
                direct backward-Euler form with coefficients b0, b1, b2.

        Returns:
            ct.TransferFunction: Discrete transfer function of the PID controller.
        """
        
        if text_option=="filtered" :
            N=50
            P=ct.TransferFunction([self.kp],[1],self.dt)
            I=ct.TransferFunction([self.ki*self.dt],[1,-1],self.dt)
            D=ct.TransferFunction([self.kd*N,-self.kd*N],[N*self.dt+1,-1],self.dt)
            PID=P+I+D
            return PID
           
        else:
            b0=self.kp+self.ki*self.dt+self.kd/self.dt
            b1=-self.kp-(2*self.kd/self.dt)
            b2=self.kd/self.dt
            PID=ct.TransferFunction([b0,b1,b2],[1,-1],self.dt)
            return PID
    def As_RST(self):
        """Converts the PID transfer function into its respective R,S and T polynomials
        """
        num = self.transferFunction.num_list[0][0]
        den = self.transferFunction.den_list[0][0]
        self.R = ct.tf(num, [1], self.dt)
        self.T = ct.tf(num, [1], self.dt)
        self.S = ct.tf(den, [1], self.dt)
        return