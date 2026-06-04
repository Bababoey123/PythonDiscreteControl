import numpy as np
import control as ct

from Models.BallBeam.StateSpace import StateSpaceModel
from Metrics_Plotting.SimLog import SimLog


class TFSimulator:
    """Calculates steps of a discrete transfer functions
    """
    ## simulates the trasfer function using difference:
    def __init__(self,tf,X_0):
        """_summary_

        Args:
            tf (ct.tf()): The discrete transfer function genrated by python-control
            X_0 (np.array([[x],[y]])): the initial state of the plant
        """
        self.num_dis=np.asarray(tf.num_list[0][0])
        self.den_dis=np.asarray(tf.den_list[0][0])

        self.u_hist = np.zeros(len(self.num_dis))
        self.y_hist = np.zeros(len(self.den_dis)-1)
        y0 = np.asarray(X_0).reshape(-1)[0]
        self.y_hist[0] = float(y0)
       

        #self.y_hist[0]=y_0
    def reset(self,X_0):
        """Resets the states of the plant, useful for a new simulation using the same plant 

        Args:
            X_0 (np.array([[x],[y]])): the initial state of the plant
        """
        self.u_hist = np.zeros(len(self.num_dis))
        self.y_hist = np.zeros(len(self.den_dis)-1)
        y0 = np.asarray(X_0).reshape(-1)[0]
        self.y_hist[0] = float(y0)
       
    
        return
    def step(self,u):
        """Calculates the next step y[k](y[k-1],y[k-2],....,u[k],u[k-1],...)

        Args:
            u (float): the input of the Transfer function

        Returns:
            y(float):the output of the Transfer function 
        """
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
class HybridSim:
    """Provides the tools for the setup and simulation of a discrete controller and a coninuous plant
    """
    def __init__(self,StateSpaceModel:StateSpaceModel,controller ,config_file):
        ## for continuous integration
        """Initialises the simulator with the continuous sim and discrete transfer function of the controller, needs improvement for RST

        Args:
            StateSpaceModel (StateSpaceModel): Contains the matrices specific for the system simulated
            controller (_type_): A controller must have a reference internal varaible and discrete transfer functions 
            config_file (_type_): The configurationfile for the system, has the simulation time as T and sampling time dt
        """
        self.dt_plant=1e-3
        ## state space models
        self.A=StateSpaceModel.A.astype(float)
        self.B=StateSpaceModel.B.astype(float)
        self.C=StateSpaceModel.C.astype(float)
        ##
        self.config_file=config_file
        ##
        self.controller=controller
        self.controller_sim=TFSimulator(self.controller.PID_TF_dis,0)
        return
    def run_continuous_control_loop(self,X_0,Logger:SimLog)->SimLog:
        """Runs the closed loop continuous simulation 

        Args:
            X_0 (_type_): Initial state
            Logger (SimLog): an instance of the SimLog class, preferably new or empty 

        Returns:
            SimLog: the logger containing the results of the simulation 
        """
        N_substep=int(self.config_file.dt/self.dt_plant) ## number of substeps between each controller update
        X=np.asarray(X_0,dtype=float)
        
        t = 0.0
        dt_control = self.config_file.dt
        u = np.array([[0.0]], dtype=float) #initial control input
    
        while t<self.config_file.T:
            y_k=self.C @ X
            u_k=np.array([[self.controller_sim.step(self.controller.reference-y_k)]])
            
            for i in range (N_substep):
                X=self.rk4_step(X,u_k)
                ## update time 
                t+=self.dt_plant
                Logger.log(t,X[0][0],u)
            if t >= self.config_file.T:
                break
                
        return Logger
    def run_impulse_respone(self,X_0,Logger:SimLog)->SimLog:
        """Runs the impulse response of the open loop plant

        Args:
            X_0 (_type_): Initial state
            Logger (SimLog): an instance of the SimLog class, preferably new or empty 

        Returns:
            SimLog: the logger containing the results of the simulation 
        """
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
        """Runs the step response of the open loop plant

        Args:
            X_0 (_type_): Initial state
            Logger (SimLog): an instance of the SimLog class, preferably new or empty 

        Returns:
            SimLog: the logger containing the results of the simulation 
        """ 
        
        X=np.asarray(X_0,dtype=float)
        
        t = 0.0
        u = np.array([[1.0]], dtype=float) #initial control input
    
        while t<self.config_file.T:
            X=self.rk4_step(X,u)
            ## update time 
            t+=self.dt_plant
            Logger.log(t,X[0][0],u)
            if t >= self.config_file.T:
                 break
                
        return Logger
    def rk4_step(self,X,u_k):
        """Runge-Kutta 4 solver 

        Args:
            X (_type_): X[k-1]
            u_k (_type_): input of the system 
        """
        
        def function(X):
            """space state derivative calculation 

            Args:
                X (_type_): the vector

            Returns:
                _type_: the derivative
            """
            return self.A @ X + self.B @ u_k 
        k1=function(X)
        k2=function(X+self.dt_plant*0.5*k1)
        k3=function(X+0.5*self.dt_plant*k2)
        k4=function(X+self.dt_plant*k3)
        
        return X +self.dt_plant/6*(k1+2*k2+2*k3+k4)
        