from ..Models.BallBeam import ballbeam_config
from ..Models.BallBeam.StateSpace import StateSpaceModel
from ..Models.BallBeam.TransferFunctions import TransferFunctionModel

from ..Simulation.simulation import TFSimulator
from ..Simulation.simulation import HybridSim

from ..Metrics_Plotting.SimLog import SimLog

import numpy as np
import control as ct


def run_discrete_control(
    plant_sim: TFSimulator,
    controller_sim: TFSimulator,
    config_file,
    r: float,
    y_0: float,
    logger: SimLog
) -> SimLog:

    N = int(config_file.T / config_file.dt)

    u = 0.0
    y = y_0

    plant_sim.reset(y_0)
    controller_sim.reset(0)

    for k in range(N):

        y = plant_sim.step(u)
        e = r - y
        u = controller_sim.step(e)

        logger.log(k * config_file.dt, y, u)

    return logger
def run_discrete_impulse_response(
    plant_sim: TFSimulator,
    config_file,
    y_0: float,
    logger: SimLog
) -> SimLog:

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
    Logger:SimLog
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
    controller_sim=TFSimulator(controller.transferFunction,0)
    N_substep=int(HybridSim.config_file.dt/HybridSim.dt_plant) ## number of substeps between each controller update
    X=np.asarray(X_0,dtype=float)
        
    t = 0.0
    dt_control = HybridSim.config_file.dt
    u = np.array([[0.0]], dtype=float) #initial control input
    
    while t<HybridSim.config_file.T:
        y_k=HybridSim.C @ X
        u_k=np.array([[controller_sim.step(controller.reference-y_k)]])
            
        for i in range (N_substep):
            X=HybridSim.rk4_step(X,u_k)
            ## update time 
            t+=HybridSim.dt_plant
            Logger.log(t,X[0][0],u)
            
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