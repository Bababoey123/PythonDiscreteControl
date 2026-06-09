##Ball and BeamSystem
## config file with all physical parameters
"""Contains all the internal parameters of the system and the simulation time variables 
"""

## physical parameters based on https://ctms.engin.umich.edu/CTMS/index.php?example=BallBeam&section=ControlStateSpace
m = 0.111 #kg
R = 0.015 #m
g = -9.8 #m/s^2
J = 9.99e-6 # kg*m^2
d= 0.03
L=1.0

## sampling frequency for the controller 
dt=1/100 #50ms 
## simulation time 
T=3#sec
H =  -m*g*d/L/(J/R**2+m)
