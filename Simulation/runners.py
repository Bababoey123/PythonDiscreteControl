from Models.BallBeam import ballbeam_config
from Models.BallBeam.StateSpace import StateSpaceModel
from Models.BallBeam.TransferFunctions import TransferFunctionModel

from Simulation.simulation import TFSimulator
from Simulation.simulation import HybridControlLoop

from Metrics_Plotting.SimLog import SimLog

import numpy as np
import control as ct


def run_discrete_control_sim(
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

        if k == 0:
            print("y0:", y)
            print("e0:", e)
            print("u0:", u)

        logger.log(k * config_file.dt, y, u)

    return logger
def run_impulse_response_sim(
    plant_sim: TFSimulator,
    config_file,
    y_0: float,
    logger: SimLog
) -> SimLog:

    N = int(config_file.T / config_file.dt)

    u = 0.0
    y = y_0

    plant_sim.reset(y_0)

    for k in range(N):
        
        u = 1.0/config_file.dt if k == 0 else 0.0
        y = plant_sim.step(u)
        if k == 0:
            print("y1:", y)
            print("u0:", u)
        

        logger.log(k * config_file.dt, y, np.array([u]))

    return logger
def run_step_response_sim(
    plant_sim: TFSimulator,
    config_file,
    y_0: float,
    logger: SimLog
) -> SimLog:

    N = int(config_file.T / config_file.dt)

    u = 1.0
    y = y_0

    plant_sim.reset(y_0)

    for k in range(N):
        
    
        y = plant_sim.step(u)
       
        logger.log(k * config_file.dt, y, np.array([u]))

    return logger
