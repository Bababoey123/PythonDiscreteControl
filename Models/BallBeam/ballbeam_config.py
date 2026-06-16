"""Ball-and-beam system physical parameters and simulation settings.

All values are taken from the CTMS reference model:
https://ctms.engin.umich.edu/CTMS/index.php?example=BallBeam&section=ControlStateSpace

Physical constants

m : float
    Ball mass [kg].
R : float
    Ball radius [m].
g : float
    Gravitational acceleration [m/s²].
J : float
    Ball moment of inertia.
d : float
    Lever arm from the beam pivot to the servo attachment point [m].
L : float
    Total beam length [m].
H : float
    Linearised plant gain.

Simulation settings

dt : float
    Controller sampling period [s].  dt = 1/50 = 0.02 s => 50 Hz sampling rate.
T : float
    Total simulation duration [s].
"""

m = 0.111     # kg  — ball mass
R = 0.015     # m   — ball radius
g = -9.8      # m/s^2 — gravitational acceleration (negative: downward)
J = 9.99e-6   # kg·m^2 — ball moment of inertia (solid sphere approximation)
d = 0.03      # m   — lever arm: pivot to servo attachment point
L = 1.0       # m   — total beam length

dt = 1 / 50   # s   — controller sampling period (50 Hz)
T = 3         # s   — total simulation duration

# Linearised plant gain
H = -m * g * d / L / (J / R**2 + m)

import numpy as np

# Continuous transfer function H(s) = H / s²
num_cont = [H]          # numerator coefficients
den_cont = [1, 0, 0]   # denominator coefficients (double integrator)

# Continuous state-space matrices (double integrator):  ẋ = A x + B u,  y = C x
A_mat = np.array([[0.0, 1.0],
                  [0.0, 0.0]])
B_mat = np.array([[0.0],
                  [H]])
C_mat = np.array([[1.0, 0.0]])
D_mat = np.array([[0.0]])
