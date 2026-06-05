"""
Hybrid Control and Simulation Library for Discrete–Continuous Systems

This library provides a unified framework for simulating control systems that combine
discrete-time controllers with continuous-time or discretized plant models.

The main goal is to support consistent comparison between:
- Pure discrete-time transfer function simulations (z-domain models)
- Continuous-time state-space simulations with numerical integration
- Hybrid closed-loop systems where a discrete controller interacts with a continuous plant

Core Features
-------------
1. Discrete-time signal simulation
   - Finite difference implementation of transfer functions
   - Used for controllers (PID, RST) and discrete plant models

2. Continuous-time plant simulation
   - State-space representation (A, B, C matrices)
   - Numerical integration (RK4)
   - Used for high-fidelity plant dynamics

3. Hybrid simulation environment
   - Discrete controller evaluated at fixed sampling time
   - Continuous plant updated at a finer integration timestep
   - Emulates real digital control hardware interacting with physical systems

4. Control architectures
   - PID controllers implemented as discrete transfer functions
   - RST controllers using polynomial representations (R, S, T)
   - Unified interface for future controller extensions (LQR, MPC, etc.)

Architecture Overview
---------------------
The library is structured around three main abstraction layers:

- Simulation layer:
  Implements numerical evolution of systems:
  TFSimulator, HybridControlLoop

- Control layer:
  Implements discrete controllers:
  DiscretePID, RSTController

- Models layer:
  Contains the configuration file with the simulations parameters (dt, the sampling time and T, the simulations lengh)
  Defines plant representations:
  StateSpaceModel, TransferFunctionModel
  
- Metrics_Plotting layer:
  Used to log the responses of the simulations, will also include different poltting functions  
  
- Utils 
    Contains helper functions such as a CSV file generator (will also contain a RST controller maker)

Design Philosophy
------------------
The library is designed to ensure consistency between:
- Analytical control design (transfer functions, z-domain algebra)
- Numerical simulation (difference equations and integration methods)
- Hybrid digital implementations (realistic sampled control loops)

A key objective is to make discrepancies between theoretical models,
numerical simulation, and hybrid execution explicit and measurable,
rather than hidden.

This allows systematic validation of:
- Discretization effects
- Sampling time influence
- Numerical integration errors
- Controller implementation consistency

"""