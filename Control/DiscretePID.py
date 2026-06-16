import numpy as np
import control as ct

from Simulation.simulation import TFSimulator


class DiscretePID:
    """Discrete PID controller discretised with backward Euler.

    Supports filtered and unfiltered derivative modes.  Does not implement output
    saturation (saturation is handled by the runner functions).  Also exposes the
    equivalent RST polynomial representation for interoperability with ``RSTController``.
    """

    def __init__(self, kp, ki, kd, dt, text_option: str = "NotFiltered"):
        """Creates an instance of the PID controller and builds its internal transfer function.

        Args:
            kp (float): Proportional gain.
            ki (float): Integral gain.
            kd (float): Derivative gain.
            dt (float): Sampling period [s], taken from the plant configuration file.
            text_option (str, optional): ``"filtered"`` enables a first-order derivative
                filter (see ``As_TransferFunction``); any other value uses the direct
                backward-Euler form.  Defaults to ``"NotFiltered"``.
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt

        self.transferFunction = self.As_TransferFunction(text_option)
        self.controller_sim = TFSimulator(self.transferFunction, 0)

        self.As_RST()

    def reset(self):
        """Resets the controller's internal state to zero.

        Call this before each new simulation run to clear accumulated integral and
        derivative history from the previous run.
        """
        self.controller_sim.reset(0)

    def setReference(self, r):
        """Sets the reference setpoint for the controller.

        Args:
            r (float): Desired plant output (setpoint).
        """
        self.reference = r

    def step(self, y):
        """Computes the control output u[k] for the current measurement y[k].

        Computes the error e[k] = r − y[k] and passes it through the PID transfer
        function implemented as a difference equation in ``TFSimulator``.

        Args:
            y (float): Current plant output (measured value).

        Returns:
            float: Control signal u[k].
        """
        e = self.reference - y
        return self.controller_sim.step(e)

    def As_TransferFunction(self, text_option: str):
        """Builds the discrete PID transfer function (position form, backward Euler).

        Both modes share the same P and I terms (backward Euler, s => (z−1)/(dt·z)):
            P(z) = Kp
            I(z) = Ki·dt·z / (z − 1)

        ``"filtered"``
            Derivative with first-order low-pass filter (N = 50):
            D(z) = Kd·N·(z − 1) / ((N·dt + 1)·z − 1)
            Filter pole at z = 1/(1 + N·dt) — always inside the unit circle.

        Any other value (default ``"NotFiltered"``)
            Pure backward-Euler derivative (no filter):
            D(z) = Kd·(z − 1) / (dt·z)

        Args:
            text_option (str): ``"filtered"`` or anything else.

        Returns:
            ct.TransferFunction: Discrete PID transfer function.
        """
        P = ct.TransferFunction([self.kp], [1], self.dt)
        I = ct.TransferFunction([self.ki * self.dt, 0], [1, -1], self.dt)

        if text_option == "filtered":
            N = 50
            D = ct.TransferFunction([self.kd * N, -self.kd * N], [N * self.dt + 1, -1], self.dt)
        else:
            D = ct.TransferFunction([self.kd / self.dt, -self.kd / self.dt], [1, 0], self.dt)

        return P + I + D

    def As_RST(self):
        """Converts the PID transfer function into equivalent R, S, T polynomials.

        For a unity-feedback PID the RST representation is:
            R(z) = T(z) = numerator of C(z)
            S(z) = denominator of C(z)
        This lets a PID be used anywhere an ``RSTController`` is expected.
        """
        num = self.transferFunction.num_list[0][0]
        den = self.transferFunction.den_list[0][0]
        self.R = ct.tf(num, [1], self.dt)
        self.T = ct.tf(num, [1], self.dt)
        self.S = ct.tf(den, [1], self.dt)
