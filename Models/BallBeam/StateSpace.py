import numpy as np
import control as ct

class StateSpaceModel:
    """Contains the specific continuous and space-state matrices of the plant 
    """
    def __init__(self,config_file):
        self.A=np.array([[0,1],[0,0]])
        self.B=np.array([[0],[config_file.H]])
        self.C=np.array([[1,0]])
        self.D=np.array([0])

        BBS_continuous=ct.StateSpace(self.A,self.B,self.C,self.D)
        BBS_discrete=BBS_continuous.sample(config_file.dt,'zoh')
        
        self.Ad=BBS_discrete.A
        self.Bd=BBS_discrete.B
        self.Cd=BBS_discrete.C
        self.Dd=BBS_discrete.D

        return