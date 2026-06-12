"""Ball-and-beam system physical parameters and simulation settings.

All values are taken from the CTMS reference model:
https://ctms.engin.umich.edu/CTMS/index.php?example=BallBeam&section=ControlStateSpace

Physical constants
------------------
m : float
    Ball mass [kg].
R : float
    Ball radius [m].
g : float
    Gravitational acceleration [m/s²] — negative so that gravity acts downward in the equations.
J : float
    Ball moment of inertia about its own centre [kg·m²], approximated as a solid sphere:
    J ≈ 2/5 · m · R² = 9.99e-6 kg·m².
d : float
    Lever arm from the beam pivot to the servo attachment point [m].
L : float
    Total beam length [m].
H : float
    Linearised plant gain [m·s⁻² per unit control input], derived from the small-angle
    approximation of the ball-on-beam acceleration:  r̈ ≈ H · (d/L) · u.
    Formula: H = −m·g·(d/L) / (J/R² + m)
    The factor (J/R² + m) is the effective translational inertia (ball mass plus rotational inertia
    mapped to the translational direction). The negative g with the sign convention gives a
    positive H: a positive servo command tilts the beam so the ball accelerates outward.

Simulation settings
-------------------
dt : float
    Controller sampling period [s].  dt = 1/50 = 0.02 s  →  50 Hz sampling rate.
T : float
    Total simulation duration [s].
"""

m = 0.111     # kg  — ball mass
R = 0.015     # m   — ball radius
g = -9.8      # m/s² — gravitational acceleration (negative: downward)
J = 9.99e-6   # kg·m² — ball moment of inertia (solid sphere approximation)
d = 0.03      # m   — lever arm: pivot to servo attachment point
L = 1.0       # m   — total beam length

dt = 1 / 50   # s   — controller sampling period (50 Hz)
T = 3         # s   — total simulation duration

# Linearised plant gain: r̈ ≈ H · α, where α = (d/L)·u is the beam tilt angle.
# H = −m·g·(d/L) / (J/R² + m)
H = -m * g * d / L / (J / R**2 + m)
