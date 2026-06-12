"""Discrete controller implementations.

Every controller in this package must expose the following interface so that it can be
used interchangeably with all runner functions in ``Simulation/runners.py``:

    controller.reset()           — clear internal state before a new simulation run
    controller.setReference(r)   — set the reference (setpoint)
    u = controller.step(y)       — compute the control output for the current measurement y

Available controllers
---------------------
DiscretePID   — PID controller discretised with backward Euler (filtered or unfiltered derivative).
RSTController — Two-degree-of-freedom polynomial RST controller.
"""
