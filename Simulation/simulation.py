import numpy as np
import control as ct

from Models.BallBeam.StateSpace import StateSpaceModel
from Metrics_Plotting.SimLog import SimLog

class Simulator:
    def __init__(self,model,X0):
        self.model=model ## in state space or the transfer function
        self.X=X0
        self.u_hist = np.zeros(len(self.model.num_dis))
        self.y_hist = np.zeros(len(self.model.den_dis)-1)
        self.y_hist[0]=self.X[0,0]       
        return
    def step(self,u):
        ## takes state space or tranfer function 
        if isinstance(self.model,StateSpaceModel):
            self.X=self.model.Ad @ self.X + self.model.Bd @ u
            y=self.model.Cd @ self.X 
            return y
        elif self.model.Tf_dis.isdtime:
            self.u_hist[1:] = self.u_hist[:-1]
            # all my elements exept the first one<= all my elements exept the last one 
            self.u_hist[0] = u
            #store first element

            y = np.dot(self.model.num_dis,self.u_hist)
            # dot product between the coeficients list and the u hist list 
            # implements b0u[k]+b1u[k-1].....
            y -= np.dot(self.model.den_dis[1:], self.y_hist)
            # dot product between the coeficients list and the u hist list 
            # implements a1y[k-1]+b2u[k-2].....

            y /= self.model.den_dis[0] # divide by a0 to get y[k]

            self.y_hist[1:] = self.y_hist[:-1]
            #all my elements exept the first one<= all my elements exept the last one 
            self.y_hist[0] = y
            #strore first element

            return y
class TFSimulator:
    ## simulates the trasfer function using difference:
    def __init__(self,tf,X_0):
        self.num_dis=np.asarray(tf.num_list[0][0])
        self.den_dis=np.asarray(tf.den_list[0][0])

        self.u_hist = np.zeros(len(self.num_dis))
        self.y_hist = np.zeros(len(self.den_dis)-1)
        y0 = np.asarray(X_0).reshape(-1)[0]
        self.y_hist[0] = float(y0)
       

        #self.y_hist[0]=y_0
    def reset(self,X_0):
        self.u_hist = np.zeros(len(self.num_dis))
        self.y_hist = np.zeros(len(self.den_dis)-1)
        y0 = np.asarray(X_0).reshape(-1)[0]
        self.y_hist[0] = float(y0)
       
    
        return
    def step(self,u):
        self.u_hist[1:] = self.u_hist[:-1]
        # all my elements exept the first one<= all my elements exept the last one 
        self.u_hist[0] = float(u)
        #strore first element

        y = np.dot(self.num_dis,self.u_hist)
        # dot product between the coeficients list and the u hist list 
        # implements b0u[k]+b1u[k-1].....
        y -= np.dot(self.den_dis[1:], self.y_hist)
        # dot product between the coeficients list and the u hist list 
        # implements a1y[k-1]+b2u[k-2].....

        y /= self.den_dis[0] # divide by a0 to get y[k]

        if len(self.y_hist) > 0:
            self.y_hist[1:] = self.y_hist[:-1]
            self.y_hist[0] = y

        
        #all my elements exept the first one<= all my elements exept the last one 
        
        #strore first element

        return y
class HybridControlLoop:
    def __init__(self,StateSpaceModel:StateSpaceModel,controller,config_file):
        ## for continuous integration
        self.dt_plant=1e-5
        ## state space models
        self.A=StateSpaceModel.A.astype(float)
        self.B=StateSpaceModel.B.astype(float)
        self.C=StateSpaceModel.C.astype(float)
        ##
        self.config_file=config_file
        ##
        self.controller=controller
        return
    def run_continuous_control_loop(self,X_0,Logger:SimLog)->SimLog:
        N_substep=int(self.config_file.dt/self.dt_plant) ## number of substeps between each controller update
        X=np.asarray(X_0,dtype=float)
        
        t = 0.0
        dt_control = self.config_file.dt
        u = np.array([[0.0]], dtype=float) #initial control input
    
        while t<self.config_file.T:
            y=self.C @ X
            u=self.controller.compute(y)
            
            for i in range (N_substep):
                x_dot=self.A @ X + self.B @ u
                ## simple forward euler 
                X+= self.dt_plant*x_dot
                y=self.C @ X
                ## update time 
                t+=self.dt_plant
                Logger.log(t,y,u)
            if t >= self.config_file.T:
                break
                
        return Logger
    def run_impulse_respone(self,X_0,Logger:SimLog)->SimLog:
        
        X=np.asarray(X_0,dtype=float)
        
        t = 0.0
        u = np.array([[0.0]], dtype=float) #initial control input
    
        while t<self.config_file.T:
            ## impulse u
            if t==0: u =np.array([[1.0]], dtype=float)
            else: u=np.array([[0.0]], dtype=float)
            ##
            x_dot=self.A @ X + self.B @ u
            ## simple forward euler 
            X+= self.dt_plant*x_dot
            y=self.C @ X
            ## update time 
            t+=self.dt_plant
            Logger.log(t,y,u)
            if t >= self.config_file.T:
                 break
                
        return Logger
    def run_step_response(self,X_0,Logger:SimLog)->SimLog:
        
        X=np.asarray(X_0,dtype=float)
        
        t = 0.0
        u = np.array([[1.0]], dtype=float) #initial control input
    
        while t<self.config_file.T:
            
            x_dot=self.A @ X + self.B @ u
            ## simple forward euler 
            X+= self.dt_plant*x_dot
            y=self.C @ X
            ## update time 
            t+=self.dt_plant
            Logger.log(t,y,u)
            if t >= self.config_file.T:
                 break
                
        return Logger

        
        