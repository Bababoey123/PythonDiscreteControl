"""Abstract base classes for plant models.

Any new plant must implement these methods so that the Simulation layer
remains indeferent for the plant.
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseNonlinearModel(ABC):
    """Abstract base for nonlinear plant models.

    Subclasses must:
    - Set ``self.C`` in ``__init__``.
    - Implement ``f(X, u)`` returning the state derivative Ẋ.
    """

    C: np.ndarray

    @abstractmethod
    def f(self, X: np.ndarray, u: float) -> np.ndarray:
        """State derivative Ẋ = f(X, u).

        Args:
            X: State vector, shape (n, 1).
            u: Scalar control input.

        Returns:
            State derivative, same shape as X.
        """


class BaseLinearStateSpaceModel:
    """Base class for linearised continuous/discrete state-space plant models.

    Subclasses must populate the following attributes in ``__init__``:

    Continuous:
        A, B, C, D — system matrices (numpy arrays)

    Discrete (ZOH):
        Ad, Bd, Cd, Dd — discretised system matrices (numpy arrays)
    """

    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray

    Ad: np.ndarray
    Bd: np.ndarray
    Cd: np.ndarray
    Dd: np.ndarray
