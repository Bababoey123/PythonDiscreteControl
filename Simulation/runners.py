"""High-level simulation runner functions.

Each runner wires together a plant simulator, an optional controller, and a SimLog, then
returns the populated logger.  Knowing the signature and side-effects of each runner is
the primary interface point for users of this library.

Hardcoded values shared by all runners
---------------------------------------
Disturbance magnitude : 2.0 (same units as the control input)
    A constant step disturbance of 2.0 is added to the plant input for all samples after
    t = 3 s.  The plant receives ``u + d`` while the logged control signal is the raw
    controller output.  Because RST enforces S(1) = 0 (integrator in S), the steady-state
    position error is exactly zero.  A PID without integral action (Ki = 0) settles with a
    permanent offset of approximately D * S(1) / R(1) ≈ D / Kp.

Disturbance onset : t > 3.0 s
    The first 3 s are reserved for the reference-tracking transient.  The disturbance is
    injected after this window so that tracking and rejection performance can be assessed
    independently from a single simulation run.

Saturation limits : ±10 (same units as the control input)
    The control signal is clipped to [−10, +10].  These limits approximate the physical
    saturation of the ball-and-beam servo actuator (roughly ±10° of beam tilt in the
    normalised model).  Saturation prevents integrator wind-up from driving the ball
    off the beam during large transients.
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

    The plant is modelled as a discrete transfer function stepped via ``TFSimulator``.
    The controller is sampled at every step (one call to ``controller.step`` per
    ``config_file.dt`` interval).

    Hardcoded values
    ----------------
    Disturbance magnitude : 2.0
        A step disturbance of 2.0 is added to the plant input ``u + d`` for all samples
        ``k >= int(3.0 / dt)``.  The logged control signal is the raw controller output.
        RST (S(1) = 0) returns the position to ``r`` exactly; PID (Ki = 0) settles with
        a permanent offset ≈ D / Kp.  See the module docstring for the rationale.
    Saturation limits : [−10, +10]
        The raw controller output is clipped to this range before being sent to the plant.
        See the module docstring for the rationale.

    Args:
        plant_sim (TFSimulator): Simulator initialised with the plant's discrete transfer function.
        controller: Controller object exposing ``step(y)`` and ``setReference(r)`` methods.
        config_file: Plant configuration module; must expose ``T`` (total time [s]) and ``dt``
            (sampling period [s]).
        r (float): Reference (setpoint) for the controller.
        y_0 (float): Initial plant output.
        logger (SimLog): SimLog instance used to record time, output, and input.
        Disturb (bool, optional): If True, adds a step input disturbance of 2.0 to the
            plant input for all samples from t = 3 s onward.  Defaults to True.
        Saturate (bool, optional): If True, clips the control signal to [−10, +10].
            Defaults to True.

    Returns:
        SimLog: The logger passed as input, now populated with simulation data.
    """
    N = int(config_file.T / config_file.dt)
    controller.reset()
    controller.setReference(r)
    u = 0.0
    y = y_0

    k_step = int(3.0 / config_file.dt)
    plant_sim.reset(y_0)
    for k in range(N):
        d = 2.0 if (Disturb and k >= k_step) else 0.0
        u = controller.step(y)
        if Saturate:
            u = float(np.clip(u, -10.0, +10.0))
        y_plant = plant_sim.step(u + d)  # disturbance at plant input
        y = y_plant

        logger.log((k + 1) * config_file.dt, np.array([y_plant]), np.array([u]))

    return logger


def run_discrete_impulse_response(
    plant_sim: TFSimulator,
    config_file,
    y_0: float,
    logger: SimLog
) -> SimLog:
    """Runs the open-loop discrete impulse response and returns the populated logger.

    The impulse magnitude is 1/dt so that the discrete impulse approximates a continuous
    Dirac delta with unit area (∫u dt ≈ (1/dt)·dt = 1).  After the first sample, u[k] = 0
    for all remaining steps.

    Hardcoded values
    ----------------
    Impulse magnitude : 1/config_file.dt
        Chosen to normalise the impulse energy to 1, consistent with the continuous-domain
        convention.  For dt = 0.02 s this gives u[0] = 50.

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
    for k in range(0, N):
        u = 1.0 / config_file.dt if k == 0 else 0.0
        y = plant_sim.step(u)
        logger.log((k + 1) * config_file.dt, np.array([y]), np.array([u]))

    return logger


def run_discrete_step_response(
    plant_sim: TFSimulator,
    config_file,
    y_0: float,
    logger: SimLog
) -> SimLog:
    """Runs the open-loop discrete step response and returns the populated logger.

    A constant unit input u = 1.0 is applied for the full simulation duration.

    Hardcoded values
    ----------------
    Step magnitude : 1.0
        Unit step input applied at k = 0 and held constant.  Scale the result by any
        other magnitude to obtain the step response for a different input level.

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
    for k in range(0, N):
        y = plant_sim.step(u)
        logger.log((k + 1) * config_file.dt, np.array([y]), np.array([u]))

    return logger


def run_continuous_control_loop(
    Simulator,
    controller,
    r,
    X_0,
    Logger: SimLog,
    Disturb=True,
    Saturate=True
) -> SimLog:
    """Runs the closed-loop hybrid simulation (continuous plant, discrete controller).

    The plant is integrated at ``dt_plant`` = 1 ms using RK4; the controller output is
    updated every ``config_file.dt`` seconds (zero-order hold between updates).  Within
    each control period, ``N_substep = config_file.dt / dt_plant`` RK4 steps are taken.
    Every RK4 substep is logged, so the output log has a 1 ms resolution regardless of
    the controller sampling period.

    Hardcoded values
    ----------------
    Disturbance magnitude : 2.0
        A step disturbance of 2.0 is added to the controller output before it is sent
        to the plant for all control periods starting at t = 3 s.  The plant receives
        ``u + d`` via ``u_k``; the logged control signal is the raw controller output.
        RST (S(1) = 0) returns to ``r`` exactly; PID (Ki = 0) settles with a permanent
        offset ≈ D / Kp.  See the module docstring for the rationale.
    Saturation limits : [−10, +10]
        The controller output is clipped to this range before the ZOH hold.
        See the module docstring for the rationale.

    Args:
        Simulator (HybridSim or NonLinearHybridSim): Hybrid simulator initialised with
            the plant's state-space model.  Must expose ``C``, ``config_file``, ``dt_plant``,
            and ``rk4_step``.
        controller: Controller object exposing ``step(y)`` and ``setReference(r)`` methods.
        r: Reference (setpoint) for the controller.
        X_0: Initial state vector of the continuous plant.
        Logger (SimLog): SimLog instance used to record time, output, and input.
        Disturb (bool, optional): If True, adds a step input disturbance of 2.0 to the
            plant input from t = 3 s onward.  Defaults to True.
        Saturate (bool, optional): If True, clips the control signal to [−10, +10].
            Defaults to True.

    Returns:
        SimLog: The logger passed as input, now populated with simulation data.
    """
    controller.reset()
    controller.setReference(r)

    N_substep = int(Simulator.config_file.dt / Simulator.dt_plant)
    X = np.asarray(X_0, dtype=float)

    t = 0.0
    u_k = np.array([[0.0]], dtype=float)

    while t < Simulator.config_file.T:
        d_in = 2.0 if (Disturb and t >= 3.0) else 0.0
        y_k = float(np.squeeze(Simulator.C @ X))

        u_scalar = float(controller.step(y_k))
        if Saturate:
            u_scalar = float(np.clip(u_scalar, -10.0, +10.0))
        u_k = np.array([[u_scalar + d_in]], dtype=float)  # disturbance at plant input

        for _ in range(N_substep):
            X = Simulator.rk4_step(X, u_k)

            t += Simulator.dt_plant

            Logger.log(t, np.array([X[0][0]]), np.array([u_scalar]))

            if t >= Simulator.config_file.T:
                break

    return Logger


def run_continuous_impulse_response(HybridSim, X_0, Logger: SimLog) -> SimLog:
    """Runs the open-loop continuous impulse response using RK4 integration.

    The impulse magnitude is 1/dt_plant so that the discrete pulse approximates a continuous
    Dirac delta with unit area, consistent with ``run_discrete_impulse_response``.
    After the first integration step, u = 0 for all remaining steps.

    Hardcoded values
    ----------------
    Impulse magnitude : 1/HybridSim.dt_plant
        With dt_plant = 1e-3 s this gives u[0] = 1000, normalising the impulse energy to 1.

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


def run_continuous_step_response(HybridSim, X_0, Logger: SimLog) -> SimLog:
    """Runs the open-loop continuous step response using RK4 integration.

    A constant unit input u = 1.0 is applied for the full simulation duration.

    Hardcoded values
    ----------------
    Step magnitude : 1.0
        Unit step input held constant from t = 0.  Scale the result for other magnitudes.

    Args:
        HybridSim (HybridSim): Hybrid simulator initialised with the plant's state-space model.
        X_0: Initial state vector of the continuous plant.
        Logger (SimLog): SimLog instance used to record time, output, and input.

    Returns:
        SimLog: The logger passed as input, now populated with simulation data.
    """
    X = np.asarray(X_0, dtype=float)

    t = 0.0
    u = np.array([[1.0]], dtype=float)

    while t < HybridSim.config_file.T:
        X = HybridSim.rk4_step(X, u)
        t += HybridSim.dt_plant
        Logger.log(t, np.array([X[0][0]]), np.array([u.item()]))
        if t >= HybridSim.config_file.T:
            break

    return Logger
