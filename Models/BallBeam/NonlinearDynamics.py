"""Nonlinear dynamics of the ball-and-beam system.

The ball dynamics are modelled with the full nonlinear term sin(α) rather than
the small-angle approximation sin(α) ≈ α used in the linear model.  The mapping
from servo command to beam angle is kept linear (α = (d/L)·u), which matches the
assumption that the servo actuator is fast enough to track its reference exactly.
"""

import numpy as np


class NonlinearBallBeamModel:
    """Nonlinear ODE model of the ball-and-beam plant.

    State vector: X = [[r], [ṙ]]
        r  — ball position along the beam [m]
        ṙ  — ball velocity along the beam [m/s]

    The ball acceleration is:
        r̈ = −m·g·sin(α) / (J/R² + m)
    where α = (d/L)·u is the beam tilt angle computed from the servo command u.
    """

    def __init__(self, config_file):
        """Stores all physical parameters from the configuration module.

        Args:
            config_file: Plant configuration module; must expose ``m``, ``g``, ``J``,
                ``R``, ``d``, and ``L``.
        """
        self.m = config_file.m
        self.g = config_file.g
        self.J = config_file.J
        self.R = config_file.R
        self.d = config_file.d
        self.L = config_file.L
        self.config_file = config_file

    def f(self, X, u: float) -> np.ndarray:
        """Evaluates the nonlinear state derivative  Ẋ = f(X, u).

        The servo angle is mapped linearly from the control input:
            α = (d / L) · u
        The ball acceleration uses the full nonlinear sine term:
            r̈ = −m·g·sin(α) / (J/R² + m)

        Note: the servo-angle mapping (u → α) remains linear even though the ball
        dynamics use sin(α).  This reflects the assumption that the servo tracks its
        command perfectly and that only the mechanical ball dynamics are nonlinear.

        Args:
            X (np.ndarray): State vector of shape (2, 1): [[r], [ṙ]].
            u (float): Servo angle command (control input) [rad or normalised units].

        Returns:
            np.ndarray: State derivative [[ṙ], [r̈]] of shape (2, 1).
        """
        r_dot = X[1, 0]
        alpha = (self.d / self.L) * u                               # beam tilt angle [rad]
        r_ddot = -self.m * self.g * np.sin(alpha) / (self.J / self.R**2 + self.m)
        return np.array([[r_dot], [r_ddot]])
