"""Nonlinear dynamics of the ball-and-beam system.

"""

import numpy as np

from Models.base import BaseNonlinearModel


class NonlinearBallBeamModel(BaseNonlinearModel):
    """Nonlinearmodel of the ball-and-beam plant.
    """

    def __init__(self, config_file):
        """Stores all physical parameters from the configuration module.

        Args:
            config_file: Plant configuration module
        """
        self.m = config_file.m
        self.g = config_file.g
        self.J = config_file.J
        self.R = config_file.R
        self.d = config_file.d
        self.L = config_file.L
        self.config_file = config_file

        # Output matrix: y = C @ X extracts ball position
        self.C = np.array([[1.0, 0.0]])

    def f(self, X: np.ndarray, u: float) -> np.ndarray:
        """Evaluates the nonlinear state derivative  Xdot = f(X, u).

        

        Args:
            X (np.ndarray): State vector .
            u (float): Servo angle.

        Returns:
            np.ndarray: Xdot.
        """
        r_dot = X[1, 0]
        alpha = (self.d / self.L) * u                               # beam tilt angle [rad]
        r_ddot = -self.m * self.g * np.sin(alpha) / (self.J / self.R**2 + self.m)
        return np.array([[r_dot], [r_ddot]])
