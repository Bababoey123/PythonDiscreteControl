import numpy as np

class SimLog:
    """Contains the arrays for logging the simulations
    """
    def __init__(self):
        """Creates empty arrays to log the simulation
        """
        self.t_hist = []
        self.y_hist = []
        self.u_hist = []
        return
    def log(self,t,y,u):
        """At each steps logs the time, input and output variables

        Args:
            t (_type_): the time 
            y (_type_): output
            u (_type_): input
        """
        self.t_hist.append(t)
        self.y_hist.append(y.item())
        self.u_hist.append(u.item())

        return