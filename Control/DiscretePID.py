import numpy as np
import control as ct

class DiscretePID:
    def __init__(self,kp,ki,kd,Max_actuation,dt):
        self.kp=kp
        self.ki=ki
        self.kd=kd

    
        self.dt=dt
        self.Max_actuation=Max_actuation

        self.PID_TF,self.PID_num,self.PID_den=self.As_TransferFunction()
        self.e_hist = np.zeros(len(self.PID_num))
        self.u_hist = np.zeros(len(self.PID_den)-1)
        return
    def setReference(self,r):
        self.reference=r
        return
    def reset(self):
        self.e_hist = np.zeros(len(self.PID_num))
        self.u_hist = np.zeros(len(self.PID_den)-1)
        return

    def compute(self,y):
        #error 
        error=self.reference-y.item()

        self.e_hist[1:] = self.e_hist[:-1]
        # all my elements exept the first one<= all my elements exept the last one 
        self.e_hist[0] = float(error)
        #strore first element
        
        u = np.dot(self.PID_num,self.e_hist)
        # dot product between the coeficients list and the u hist list 
        # implements b0u[k]+b1u[k-1].....
        u-= np.dot(self.PID_den[1:], self.u_hist)
        # dot product between the coeficients list and the u hist list 
        # implements a1y[k-1]+b2u[k-2].....

        u /= self.PID_den[0] # divide by a0 to get y[k]

        if len(self.u_hist) > 0:
            self.u_hist[1:] = self.u_hist[:-1]
            self.u_hist[0] = u

        
        #all my elements exept the first one<= all my elements exept the last one 
        #strore first element
        return np.array([[u]])
    
    def As_TransferFunction(self):
        b0=self.kp+self.ki*self.dt+self.kd/self.dt
        b1=-self.kp-(2*self.kd/self.dt)
        b2=self.kd/self.dt
        PID=ct.TransferFunction([b0,b1,b2],[1,-1],self.dt)
        return PID,[b0,b1,b2],[1,-1]