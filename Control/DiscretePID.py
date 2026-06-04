import numpy as np
import control as ct

class DiscretePID:
    def __init__(self,kp,ki,kd,Max_actuation,dt,text_option:str="filtered"):
        self.kp=kp
        self.ki=ki
        self.kd=kd

    
        self.dt=dt
        self.Max_actuation=Max_actuation

        self.PID_TF_dis=self.As_TransferFunction(text_option)
        self.PID_num=np.asarray(self.PID_TF_dis.num_list[0][0])
        self.PID_den=np.asarray(self.PID_TF_dis.den_list[0][0])
        self.PID_TF_cont=ct.tf([kp**2,kp,ki],[1,0])
        
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

  
    
    def As_TransferFunction(self,text_option:str):
        
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