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
        """Builds the discrete PID transfer function using backward Euler discretisation.

        Two modes are available:

        ``"filtered"``
            Constructs the controller as P + I + D where the derivative term includes a
            first-order low-pass filter with filter coefficient N = 50.  The filter
            bandwidth is N / dt rad/s (≈ 2500 rad/s at dt = 0.02 s), which attenuates
            high-frequency noise amplified by differentiation while still providing
            adequate phase lead.  Discrete forms (backward Euler, s → (z−1)/(dt·z)):
                P(z) = kp
                I(z) = ki·dt / (z − 1)
                D(z) = kd·N·(z − 1) / ((N·dt + 1)·z − 1)

        Any other value (default ``"NotFiltered"``)
            Direct backward-Euler PID with coefficients:
                b0 = kp + ki·dt + kd/dt
                b1 = −kp − 2·kd/dt
                b2 = kd/dt
            Transfer function: C(z) = (b0·z² + b1·z + b2) / (z² − z)

        Args:
            text_option (str): Mode selector; ``"filtered"`` or anything else.

        Returns:
            ct.TransferFunction: Discrete transfer function of the PID controller.
        """
        if text_option == "filtered":
            N = 50  # derivative filter coefficient: filter bandwidth = N/dt rad/s
            P = ct.TransferFunction([self.kp], [1], self.dt)
            I = ct.TransferFunction([self.ki * self.dt], [1, -1], self.dt)
            D = ct.TransferFunction([self.kd * N, -self.kd * N], [N * self.dt + 1, -1], self.dt)
            return P + I + D
        else:
            b0 = self.kp + self.ki * self.dt + self.kd / self.dt
            b1 = -self.kp - (2 * self.kd / self.dt)
            b2 = self.kd / self.dt
            return ct.TransferFunction([b0, b1, b2], [1, -1], self.dt)

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
