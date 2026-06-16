import numpy as np
import control as ct


class TransferFunctionModel:
    """Continuous and ZOH-discretised transfer function for plant.

    The config must expose:
        ``num_cont`` — list of continuous numerator coefficients
        ``den_cont`` — list of continuous denominator coefficients
        ``dt``       — sampling period [s]
    """

    def __init__(self, config_file):
        self.Tf_cont = ct.tf(config_file.num_cont, config_file.den_cont)
        self.Tf_dis = ct.c2d(self.Tf_cont, config_file.dt, method='zoh')

        self.num_dis = np.asarray(self.Tf_dis.num_list[0][0])
        self.den_dis = np.asarray(self.Tf_dis.den_list[0][0])
