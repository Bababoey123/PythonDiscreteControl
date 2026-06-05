# Hybrid Control Simulation Library

This library provides a lightweight framework for simulating control systems combining:

- Discrete-time controllers (PID, RST)
- Continuous-time or discretized plant models
- Hybrid closed-loop systems with sampled control and continuous dynamics

## Core Components

### 1. Discrete Transfer Function Simulation
Implements digital systems using finite difference equations:
- Used for controllers and discrete plant models

### 2. State-Space Simulation
Continuous-time plant simulation using:

- State-space models (A, B, C matrices)
- Numerical integration (RK4)

### 3. Hybrid Simulation Loop
Digital controller interacting with a continuous plant:

- Controller updated at sampling time `dt`
- Plant integrated at finer timestep
- Models real embedded control systems

## Controllers

### PID Controller
- Implemented as discrete transfer function
- Supports filtered and non-filtered derivative action
- Equivalent RST polynomial representation included

### RST Structure
- R, S, T polynomials derived for now from the PID's TF
- Used for algebraic control design and analysis

