import numpy as np
import control as ct

class LinearStateSpaceModel:
    """Linearised continuous and discrete state-space model of the ball-and-beam plant.

    The continuous model is a double integrator:
        ẋ = [[0, 1], [0, 0]] x + [[0], [H]] u
        y = [1, 0] x

    Both continuous (A, B, C, D) and ZOH-discretised (Ad, Bd, Cd, Dd) matrices
    are computed at construction time.
    """
    def __init__(self, config_file):
        """Builds the state-space matrices and discretises with ZOH at the config sampling rate.

        Args:
            config_file: Plant configuration module; must expose ``H`` (linearised plant gain)
                and ``dt`` (sampling period in seconds).
        """
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

    
    