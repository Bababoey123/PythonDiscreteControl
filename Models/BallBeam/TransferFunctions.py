import numpy as np
import control as ct

class TransferFunctionModel:
    """Continuous and ZOH-discretised transfer functions of the ball-and-beam plant.

    The continuous plant is a double integrator: H(s) = H / s².
    """
    def __init__(self, config_file):
        """Builds the continuous TF and its ZOH discretisation, then extracts coefficient arrays.

        Args:
            config_file: Plant configuration module; must expose ``H`` (linearised gain)
                and ``dt`` (sampling period in seconds).
        """
        self.Tf_cont=ct.tf(config_file.H,[1,0,0])
        self.Tf_dis= ct.c2d(self.Tf_cont,config_file.dt,method='zoh')

        self.num_dis=np.asarray(self.Tf_dis.num_list[0][0])
        self.den_dis=np.asarray(self.Tf_dis.den_list[0][0])

        return
    