"""The most imporotant part of the library, contains the functions that run differen simulations, knowing the iputs and ouptus of each is mandatory to use the library
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
    Disturb=True,
    Saturate=True
) -> SimLog:
    """Runs the discrete closed-loop simulation and returns the populated logger.

    Args:
        plant_sim (TFSimulator): Simulator initialised with the plant's discrete transfer function.
        controller: Controller object exposing ``step(y)`` and ``setReference(r)`` methods.
        config_file: Plant configuration module; must expose ``T`` (total time) and ``dt``
            (sampling period).
        r (float): Reference (setpoint) for the controller.
        y_0 (float): Initial plant output.
        logger (SimLog): SimLog instance used to record time, output, and input.
        Disturb (bool, optional): If True, applies a step input disturbance of 2.0
            after t = 3 s. Defaults to True.
        Saturate (bool, optional): If True, clips the control signal to [-10, +10].
            Defaults to True.

    Returns:
        SimLog: The logger passed as input, now populated with simulation data.
    """
    
    N = int(config_file.T / config_file.dt)
    controller.setReference(r)
    u = 0.0
    y = y_0

    plant_sim.reset(y_0)
    for k in range(N):
        d = 2.0 if (Disturb and k * config_file.dt > 3.0) else 0.0
        u = controller.step(y)
        if Saturate:
            u = float(np.clip(u, -10.0, +10.0))
        y = plant_sim.step(u + d)
    
        logger.log((k+1) * config_file.dt, np.array([y]), np.array([u]))
        
    return logger

def run_discrete_impulse_response(
    plant_sim: TFSimulator,
    config_file,
    y_0: float,
    logger: SimLog
) -> SimLog:
    """Runs the open-loop discrete impulse response and returns the populated logger.

    The impulse magnitude is 1/dt so that the discrete impulse approximates a
    continuous Dirac delta with unit area.

    Args:
        plant_sim (TFSimulator): Simulator initialised with the plant's discrete transfer function.
        config_file: Plant configuration module; must expose ``T`` and ``dt``.
        y_0 (float): Initial plant output.
        logger (SimLog): SimLog instance used to record time, output, and input.

    Returns:
        SimLog: The logger passed as input, now populated with simulation data.
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
    """Runs the open-loop discrete step response and returns the populated logger.

    A constant unit input u = 1.0 is applied for the full simulation duration.

    Args:
        plant_sim (TFSimulator): Simulator initialised with the plant's discrete transfer function.
        config_file: Plant configuration module; must expose ``T`` and ``dt``.
        y_0 (float): Initial plant output.
        logger (SimLog): SimLog instance used to record time, output, and input.

    Returns:
        SimLog: The logger passed as input, now populated with simulation data.
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
    Disturb=True,
    Saturate=True
    )->SimLog:
    """Runs the closed-loop hybrid simulation (continuous plant, discrete controller).

    The plant is integrated at ``dt_plant`` (1 ms) using RK4; the controller updates
    every ``config_file.dt`` seconds.

    Args:
        HybridSim (HybridSim): Hybrid simulator initialised with the plant's state-space model.
        controller: Controller object exposing ``step(y)`` and ``setReference(r)`` methods.
        r: Reference (setpoint) for the controller.
        X_0: Initial state vector of the continuous plant.
        Logger (SimLog): SimLog instance used to record time, output, and input.
        Disturb (bool, optional): If True, applies a step input disturbance of 6.0
            after t = 3 s. Defaults to True.
        Saturate (bool, optional): If True, clips the control signal to [-10, +10].
            Defaults to True.

    Returns:
        SimLog: The logger passed as input, now populated with simulation data.
    """
    controller.setReference(r)
    
    N_substep=int(HybridSim.config_file.dt/HybridSim.dt_plant) ## number of substeps between each controller update
    X=np.asarray(X_0,dtype=float)
        
    t = 0.0
    u_k = np.array([[0.0]], dtype=float)

    while t < HybridSim.config_file.T:
        # Sample the plant output at the start of each control period
        y_k = float(np.squeeze(HybridSim.C @ X))

        # Compute and optionally saturate the control action (ZOH: held for N_substep steps)
        u_scalar = float(controller.step(y_k))
        if Saturate:
            u_scalar = float(np.clip(u_scalar, -10.0, +10.0))
        u_k = np.array([[u_scalar]], dtype=float)

        # Step the continuous plant forward one control period
        for i in range(N_substep):
            if Disturb and t > 3.0:
                d = np.array([[6.0]])
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
        """Runs the open-loop continuous impulse response using RK4 integration.

        The impulse magnitude is 1/dt_plant so that the discrete pulse approximates
        a continuous Dirac delta with unit area, consistent with run_discrete_impulse_response.

        Args:
            HybridSim (HybridSim): Hybrid simulator initialised with the plant's state-space model.
            X_0: Initial state vector of the continuous plant.
            Logger (SimLog): SimLog instance used to record time, output, and input.

        Returns:
            SimLog: The logger passed as input, now populated with simulation data.
        """
        X = np.asarray(X_0, dtype=float)
        t = 0.0
        k = 0

        while t < HybridSim.config_file.T:
            u = np.array([[1.0 / HybridSim.dt_plant]], dtype=float) if k == 0 else np.array([[0.0]], dtype=float)
            X = HybridSim.rk4_step(X, u)
            t += HybridSim.dt_plant
            k += 1
            Logger.log(t, HybridSim.C @ X, u)
            if t >= HybridSim.config_file.T:
                break

        return Logger
def run_continuous_step_response(HybridSim,X_0,Logger:SimLog)->SimLog:
        """Runs the open-loop continuous step response using RK4 integration.

        A constant unit input u = 1.0 is applied for the full simulation duration.

        Args:
            HybridSim (HybridSim): Hybrid simulator initialised with the plant's state-space model.
            X_0: Initial state vector of the continuous plant.
            Logger (SimLog): SimLog instance used to record time, output, and input.

        Returns:
            SimLog: The logger passed as input, now populated with simulation data.
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