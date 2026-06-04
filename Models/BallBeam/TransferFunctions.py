import numpy as np
import control as ct

class TransferFunctionModel:
    """Contains the discrete and continuous transfer function of the plant
    """
    def __init__(self,config_file):
        """Initialises both transfer function, also has the arrays of containin numerator and denominator coeficients

        Args:
            config_file (.py file): the config file of the model
        """
        self.Tf_cont=ct.tf(config_file.H,[1,0,0])
        self.Tf_dis= ct.c2d(self.Tf_cont,config_file.dt,method='zoh')

        self.num_dis=np.asarray(self.Tf_dis.num_list[0][0])
        self.den_dis=np.asarray(self.Tf_dis.den_list[0][0])

        return
    