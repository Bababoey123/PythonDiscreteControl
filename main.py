from Models.BallBeam import ballbeam_config
from Models.BallBeam.StateSpace import StateSpaceModel
from Models.BallBeam.TransferFunctions import TransferFunctionModel

from Control.DiscretePID import DiscretePID

from Simulation.simulation import TFSimulator
from Simulation.simulation import HybridControlLoop
from Simulation.runners import *

from Metrics_Plotting.SimLog import SimLog
from Metrics_Plotting.Plotting import Plotting

import numpy as np
import control as ct
import matplotlib.pyplot as plt


# =============================================================================
# Configuration
# =============================================================================

kp, ki, kd = 10, 0.1, 0.4
r = 1.0

X_0 = np.array([[0], [0]])

model = TransferFunctionModel(ballbeam_config)

PID = DiscretePID(
    kp,
    ki,
    kd,
    10,
    ballbeam_config.dt
)
PID.setReference(r)


plant_sim = TFSimulator(model.Tf_dis, 0)
controller_sim = TFSimulator(PID.PID_TF, 0)

# =============================================================================
# Discrete Closed Loop Simulation
# =============================================================================



logger_discrete = SimLog()

logger_discrete = run_discrete_control_sim(
    plant_sim,
    controller_sim,
    ballbeam_config,
    r,
    0,
    logger_discrete
)

plotter = Plotting()
plotter.plotAll(logger_discrete, "Discrete Simulation")


# =============================================================================
# Reference Response (python-control)
# =============================================================================

L = PID.PID_TF * model.Tf_dis
T = ct.feedback(L, 1)

t = np.arange(
    0,
    ballbeam_config.T,
    ballbeam_config.dt
)

_, y_lib = ct.step_response(T, t)


# =============================================================================
# Hybrid Simulation
# =============================================================================

PID.reset()

plant_model = StateSpaceModel(ballbeam_config)

hybrid_loop = HybridControlLoop(
    plant_model,
    PID,
    ballbeam_config
)

logger_hybrid = SimLog()

logger_hybrid = hybrid_loop.run_continuous_control_loop(
    X_0,
    logger_hybrid
)


# =============================================================================
# Comparison Plot
# =============================================================================

plt.figure()

plt.plot(t, y_lib, label="Library TF")
plt.plot(
    logger_discrete.t_hist,
    logger_discrete.y_hist,
    color='blue',
    linewidth=2,
    label="Discrete Simulation"
)
plt.plot(
    logger_hybrid.t_hist,
    logger_hybrid.y_hist,
    color='red',
    linewidth=2,
    linestyle='dashed',
    label="Hybrid Simulation"
)

plt.title("Discrete vs Library vs Hybrid")
plt.grid()
plt.legend()

plt.show()

# =============================================================================
# Impulse Respone Simulator 
# =============================================================================
impulse_logger=SimLog()
impulse_logger=run_impulse_response_sim(plant_sim,ballbeam_config,0.0,impulse_logger)

print('own')
for k in range(10):
    print(k, impulse_logger.y_hist[k])

plt.figure()
plt.plot(
    impulse_logger.t_hist,
    impulse_logger.y_hist,
    label="Impulse Sim"
)

print(model.Tf_dis)

# =============================================================================
# Impulse Respone python-control 
# =============================================================================

t = np.arange(
    0,
    ballbeam_config.T,
    ballbeam_config.dt
)

_, y_lib = ct.impulse_response(model.Tf_dis, t)

print('lib')
for k in range(10):
    print(k, y_lib[k])

plt.plot(
    t,
    y_lib,
    linestyle='dashed',
    label="Impulse Lib"
)
plt.grid()
plt.legend()
plt.show()

# =============================================================================
#  Step Response Simulator
# =============================================================================
impulse_logger=SimLog()
impulse_logger=run_step_response_sim(plant_sim,ballbeam_config,0.0,impulse_logger)

print('own')
for k in range(10):
    print(k, impulse_logger.y_hist[k])

plt.figure()
plt.plot(
    impulse_logger.t_hist,
    impulse_logger.y_hist,
    linewidth=3,
    label="Step Sim"
)

print(model.Tf_dis)

# =============================================================================
# Step Response python-control 
# =============================================================================

t = np.arange(
    0,
    ballbeam_config.T,
    ballbeam_config.dt
)

_, y_lib = ct.step_response(model.Tf_dis, t)

print('lib')
for k in range(10):
    print(k, y_lib[k])

plt.plot(
    t,
    y_lib,
    linestyle=':',
    linewidth=3,
    label="Step Lib"
)
plt.grid()
plt.title('Comparaison des réponses inditielles')
plt.legend()
plt.show()
