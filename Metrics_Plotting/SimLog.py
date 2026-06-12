import numpy as np


class SimLog:
    """Accumulates time, output, and input samples from a simulation run.

    Each call to ``log`` appends one sample.  After the simulation, ``t_hist``,
    ``y_hist``, and ``u_hist`` hold the complete trajectories as plain Python lists,
    ready for plotting or export.
    """

    def __init__(self):
        """Creates empty lists to accumulate the simulation trajectory."""
        self.t_hist = []
        self.y_hist = []
        self.u_hist = []

    def log(self, t: float, y: np.ndarray, u: np.ndarray):
        """Appends one time step to the trajectory.

        Both ``y`` and ``u`` are expected to be single-element arrays; ``.item()``
        is called to store plain Python floats rather than NumPy scalars, which
        prevents unbounded array accumulation over long runs.

        Args:
            t (float): Current simulation time [s].
            y (np.ndarray): Plant output at time t, shape (1,).
            u (np.ndarray): Control input applied at time t, shape (1,).
        """
        self.t_hist.append(t)
        self.y_hist.append(y.item())
        self.u_hist.append(u.item())
