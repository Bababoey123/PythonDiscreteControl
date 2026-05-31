import numpy as np

class SimLog:
    def __init__(self):
        self.t_hist = []
        self.y_hist = []
        self.u_hist = []
        return
    def log(self,t,y,u):
        self.t_hist.append(t)
        self.y_hist.append(y.item())
        self.u_hist.append(u.item())

        return