import numpy as np
import control as ct

class DiscretePID:
    """The discrete PID class can work in both filtered and non-filtered modes
    Discretised in backwards euler and dosent have output saturation for now  
    """
    def __init__(self,kp,ki,kd,dt,text_option:str="filtered"):
        """Creates an instance of the PID class,

        Args:
            kp (_float_): Kp represents the strengh of the proportionnal action of the controller
            ki (_float_): Ki represents the strengh of the integral action of the controller
            kd (_float_): Kd represents the strengh of the derivative action of the controller
            dt (_float_): sampling time of the controller, comes form the config file of the plant
            text_option (str, optional):Option to choose tha filtered dervative action. Defaults to "filtered".
        """
        self.kp=kp
        self.ki=ki
        self.kd=kd
        self.dt=dt

        self.transferFunction=self.As_TransferFunction(text_option)
        self.PID_num=np.asarray(self.transferFunction.num_list[0][0])
        self.PID_den=np.asarray(self.transferFunction.den_list[0][0])
        self.PID_TF_cont=ct.tf([kp**2,kp,ki],[1,0])
        
        self.e_hist = np.zeros(len(self.PID_num))
        self.u_hist = np.zeros(len(self.PID_den)-1)
        return
    def setReference(self,r):
        """Sets the reference of the PID controller

        Args:
            r (_float_): the reference (often 1.0)
        """
        self.reference=r
        return
    def reset(self):
        """resets the error and the input history of the controller, isnt really needed 
        """
        self.e_hist = np.zeros(len(self.PID_num))
        self.u_hist = np.zeros(len(self.PID_den)-1)
        return

  
    
    def As_TransferFunction(self,text_option:str):
        """Computes the coeficients of the discrete transfer function. Carefull, ct.tf takes arrays as coeficients on z^-1

        Args:
            text_option (str): the option to chosse the derivative filter

        Returns:
            The python-control transfer function object of the apropriate PID controller
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