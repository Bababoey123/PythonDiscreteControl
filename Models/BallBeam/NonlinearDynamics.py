import numpy as np
"""Model of the non-linear ball and beam system
"""

class NonlinearBallBeamModel:
    def __init__(self,config_file):
        self.m=config_file.m
        self.g=config_file.g
        self.g = config_file.g
        self.J = config_file.J
        self.R = config_file.R
        self.d = config_file.d
        self.L = config_file.L
        self.config_file=config_file

        return
    def f(self,X,u:float):
        """Non linear ODE, used in the simuation
        Note: The linearization for the servo angnle has been kept 
        """
        r_dot=X[1,0]
        alpha= (self.d/self.L)*u
        r_ddot=-self.m*self.g*np.sin(alpha)/(self.J / self.R**2 + self.m)
        return np.array([[r_dot], [r_ddot]])