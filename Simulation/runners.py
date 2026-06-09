"""The most imporotant port of the library, contains the functions that run differen simulations, knowing the iputs and ouptus of each is mandatory to use the library
"""


from Models.BallBeam import ballbeam_config
from Models.BallBeam.StateSpace import LinearStateSpaceModel
from Models.BallBeam.TransferFunctions import TransferFunctionModel

from Simulation.simulation import TFSimulator
from Simulation.simulation import HybridSim

from Metrics_Plotting.SimLog import SimLog

import numpy as np
import control as ct


def run_discrete_control(
    plant_sim: TFSimulator,
    controller,
    config_file,
    r: float,
    y_0: float,
    logger: SimLog,
    Disturb=True
) -> SimLog:
    """Computes the discrete closed loop response of the plant and it's controller, returns the logger

    Args:
        plant_sim (TFSimulator): A TFSimulator instance initalised with the plant's model
        controller (_type_): a controller object, must include step and reset methods
        config_file (_type_): configuration file of the plant, contains the simulations lengh and sampling time 
        r (float): the reference 
        y_0 (float): ititial state 
        logger (SimLog): the SimLog() logger for data recording

    Returns:
        SimLog: The initial logger, now with all the data 
    """
    
    N = int(config_file.T / config_file.dt)
    controller.setReference(r)
    u = 0.0
    y = y_0

    plant_sim.reset(y_0)
    for k in range(N):
        if Disturb and k*config_file.dt > 3.0:
                d = np.array([[2.0]])
        else:
                d = np.array([[0.0]])
        u = controller.step(y)
        y = plant_sim.step(u+d)
    
        logger.log((k+1) * config_file.dt, np.array([y]), np.array([u]))
        
    return logger

def run_discrete_impulse_response(
    plant_sim: TFSimulator,
    config_file,
    y_0: float,
    logger: SimLog
) -> SimLog:
    """Computes the discrete impulse response of the plant, returns the logger 
     Args:
        plant_sim (TFSimulator): A TFSimulator instance initalised with the plant's model
        config_file (_type_): configuration file of the plant, contains the simulations lengh and sampling time 
        y_0 (float): ititial state 
        logger (SimLog): the SimLog() logger for data recording

    Returns:
        SimLog: The entry logger, now with all the data 

   
    """

    N = int(config_file.T / config_file.dt)

    u = 0.0
    y = y_0

    plant_sim.reset(y_0)
    logger.log(0, np.array([y]), np.array([u]))
    for k in range(0,N):
        
        u = 1.0/config_file.dt if k == 0 else 0.0
        y = plant_sim.step(u)

        logger.log((k+1) * config_file.dt, y, np.array([u]))

    return logger
def run_discrete_step_response(
    plant_sim: TFSimulator,
    config_file,
    y_0: float,
    logger: SimLog
) -> SimLog:
    """Computes the discrete impulse response on the plant
     Args:
        plant_sim (TFSimulator): A TFSimulator instance initalised with the plant's model
        config_file (_type_): configuration file of the plant, contains the simulations lengh and sampling time 
        y_0 (float): ititial state 
        logger (SimLog): the SimLog() logger for data recording

    Returns:
        SimLog: The entry logger, now with all the data 
    """

    N = int(config_file.T / config_file.dt)

    u = 1.0
    y = y_0
    plant_sim.reset(y_0)
    logger.log(0, np.array([y]), np.array([u]))
    for k in range(0,N):
        
    
        y = plant_sim.step(u)
       
        logger.log((k+1) * config_file.dt, y, np.array([u]))

    return logger
def run_continuous_control_loop(
    HybridSim:HybridSim,
    controller,
    r,
    X_0,
    Logger:SimLog,
    Disturb=True
    )->SimLog:
    """Runs the cloed loop hybrid simulation

    Args:
        HybridSim (HybridSim): The HybridSim object, initialised withe the plants model
        controller: The controller object, must have a .transerFunction with his discrete tranfer function
        r: The reference for the controller
        X_0: The initial state
        Logger (SimLog): The empty SimLog() instance of the simulation
        

    Returns:
        SimLog: The SimLog instance passed as input, now with the data of the simulation
    """
    controller.setReference(r)
    
    N_substep=int(HybridSim.config_file.dt/HybridSim.dt_plant) ## number of substeps between each controller update
    X=np.asarray(X_0,dtype=float)
        
    t = 0.0
    dt_control = HybridSim.config_file.dt
    u_k = np.array([[0.0]], dtype=float) #initial control input
    
    while t < HybridSim.config_file.T:
        # 1. Force the output product to extract as a clean scalar float
        y_matrix = HybridSim.C @ X
        y_k = float(np.squeeze(y_matrix)) 
        
        # 2. Force the controller step output to be a pure float
        u_scalar = float(controller.step(y_k))
        #print("u_k-",u_scalar,"y_k-",y_k)
        # 3. Format the control action array safely for your RK4 block
        u_k = np.array([[u_scalar]], dtype=float)
            
        # 4. Step the continuous plant states forward
        for i in range(N_substep):
            if Disturb and t > 3.0:
                d = np.array([[2.0]])
            else:
                d = np.array([[0.0]])

            u_plant = u_k + d
            X = HybridSim.rk4_step(X, u_plant)
            
            # Update physical loop clock
            t += HybridSim.dt_plant
            
            # Log as pure NumPy primitives to prevent array accumulation 
            Logger.log(t, np.array([X[0][0]]), np.array([u_scalar]))
            
            if t >= HybridSim.config_file.T:
                break
    return Logger
def run_continuous_impulse_respone(HybridSim,X_0,Logger:SimLog)->SimLog:
        """Runs the impulse response of the open loop plant

        Args:
            HybridSim (HybridSim): An HybridSim instance initalised with the plants space state model
            X_0 (_type_): Initial state
            Logger (SimLog): an instance of the SimLog class, preferably new or empty 

        Returns:
            SimLog: the logger containing the results of the simulation 
        """
        X=np.asarray(X_0,dtype=float)
        
        t = 0.0
        u = np.array([[0.0]], dtype=float) #initial control input
    
        while t<HybridSim.config_file.T:
            ## impulse u
            if t==0: u =np.array([[1.0]], dtype=float)
            else: u=np.array([[0.0]], dtype=float)
            ##
            x_dot=HybridSim.A @ X + HybridSim.B @ u
            ## simple forward euler 
            X+= HybridSim.dt_plant*x_dot
            y=HybridSim.C @ X
            ## update time 
            t+=HybridSim.dt_plant
            Logger.log(t,y,u)
            if t >= HybridSim.config_file.T:
                 break
                
        return Logger
def run_continuous_step_response(HybridSim,X_0,Logger:SimLog)->SimLog:
        """Runs the step response of the open loop plant

        Args:
            HybridSim (HybridSim): An HybridSim instance initalised with the plants space state model
            X_0 (_type_): Initial state
            Logger (SimLog): an instance of the SimLog class, preferably new or empty 

        Returns:
            SimLog: the logger containing the results of the simulation 
        """ 
        
        X=np.asarray(X_0,dtype=float)
        
        t = 0.0
        u = np.array([[1.0]], dtype=float) #initial control input
    
        while t<HybridSim.config_file.T:
            
            X=HybridSim.rk4_step(X,u)
            ## update time 
            t+=HybridSim.dt_plant
            Logger.log(t,X[0][0],u)
            if t >= HybridSim.config_file.T:
                 break
                
        return Logger