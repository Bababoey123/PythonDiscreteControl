"""Tests for the generic Models layer.

Covers:
  - Backward compatibility: TF and SS models produce the same numerical results
    as the old hardcoded implementations.
  - Inheritance: BallBeam classes satisfy the base-class contracts.
  - NonlinearBallBeamModel: physics correctness of f(X, u) and presence of C.
  - NonLinearHybridSim: C is read from the model, not hardcoded.
  - Generic substitutability: mock plants accepted by both simulators.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import control as ct

from Models.BallBeam import ballbeam_config
from Models.BallBeam.TransferFunctions import TransferFunctionModel
from Models.BallBeam.StateSpace import LinearStateSpaceModel
from Models.BallBeam.NonlinearDynamics import NonlinearBallBeamModel
from Models.base import BaseNonlinearModel, BaseLinearStateSpaceModel
from Simulation.simulation import HybridSim, NonLinearHybridSim


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ref_tf_cont():
    """Reference continuous TF built the old way (hardcoded)."""
    return ct.tf([ballbeam_config.H], [1, 0, 0])


def _ref_ss_matrices():
    """Reference SS matrices built the old way (hardcoded)."""
    H = ballbeam_config.H
    A = np.array([[0, 1], [0, 0]])
    B = np.array([[0], [H]])
    C = np.array([[1, 0]])
    D = np.array([0])
    return A, B, C, D


# ---------------------------------------------------------------------------
# 1. Config completeness
# ---------------------------------------------------------------------------

def test_config_has_required_tf_fields():
    assert hasattr(ballbeam_config, 'num_cont'), "ballbeam_config missing num_cont"
    assert hasattr(ballbeam_config, 'den_cont'), "ballbeam_config missing den_cont"
    assert len(ballbeam_config.num_cont) >= 1
    assert len(ballbeam_config.den_cont) >= 1
    print("test_config_has_required_tf_fields PASSED.")


def test_config_has_required_ss_fields():
    for field in ('A_mat', 'B_mat', 'C_mat', 'D_mat'):
        assert hasattr(ballbeam_config, field), f"ballbeam_config missing {field}"
    assert ballbeam_config.A_mat.shape == (2, 2)
    assert ballbeam_config.B_mat.shape == (2, 1)
    assert ballbeam_config.C_mat.shape == (1, 2)
    assert ballbeam_config.D_mat.shape == (1, 1)
    print("test_config_has_required_ss_fields PASSED.")


# ---------------------------------------------------------------------------
# 2. TransferFunctionModel backward compatibility
# ---------------------------------------------------------------------------

def test_tf_model_continuous_matches_reference():
    model = TransferFunctionModel(ballbeam_config)
    ref = _ref_tf_cont()
    assert np.allclose(
        np.squeeze(model.Tf_cont.num_list[0][0]),
        np.squeeze(ref.num_list[0][0]),
        atol=1e-10
    ), "Continuous TF numerator mismatch"
    assert np.allclose(
        np.squeeze(model.Tf_cont.den_list[0][0]),
        np.squeeze(ref.den_list[0][0]),
        atol=1e-10
    ), "Continuous TF denominator mismatch"
    print("test_tf_model_continuous_matches_reference PASSED.")


def test_tf_model_discrete_poles_on_unit_circle():
    model = TransferFunctionModel(ballbeam_config)
    poles = model.Tf_dis.poles()
    # Double integrator discretised by ZOH → two poles exactly at z = 1
    assert np.all(np.abs(np.abs(poles) - 1.0) < 1e-6), \
        f"Expected poles on unit circle, got radii {np.abs(poles)}"
    print("test_tf_model_discrete_poles_on_unit_circle PASSED.")


def test_tf_model_num_den_arrays_extracted():
    model = TransferFunctionModel(ballbeam_config)
    assert isinstance(model.num_dis, np.ndarray)
    assert isinstance(model.den_dis, np.ndarray)
    assert model.num_dis.ndim == 1
    assert model.den_dis.ndim == 1
    print("test_tf_model_num_den_arrays_extracted PASSED.")


def test_tf_model_discrete_step_matches_reference():
    """Step response of generic model must match the hardcoded reference."""
    model = TransferFunctionModel(ballbeam_config)
    ref = _ref_tf_cont()
    ref_dis = ct.c2d(ref, ballbeam_config.dt, method='zoh')

    t = np.arange(0, 1.0, ballbeam_config.dt)
    _, y_new = ct.step_response(model.Tf_dis, t)
    _, y_ref = ct.step_response(ref_dis, t)

    assert np.allclose(y_new, y_ref, atol=1e-10), \
        f"Step response diverged (max error {np.max(np.abs(y_new - y_ref)):.2e})"
    print("test_tf_model_discrete_step_matches_reference PASSED.")


# ---------------------------------------------------------------------------
# 3. LinearStateSpaceModel backward compatibility
# ---------------------------------------------------------------------------

def test_ss_model_inherits_base():
    model = LinearStateSpaceModel(ballbeam_config)
    assert isinstance(model, BaseLinearStateSpaceModel), \
        "LinearStateSpaceModel must inherit BaseLinearStateSpaceModel"
    print("test_ss_model_inherits_base PASSED.")


def test_ss_model_continuous_matrices_match_reference():
    model = LinearStateSpaceModel(ballbeam_config)
    A_ref, B_ref, C_ref, D_ref = _ref_ss_matrices()
    assert np.allclose(model.A, A_ref, atol=1e-10), "A matrix mismatch"
    assert np.allclose(model.B, B_ref, atol=1e-10), "B matrix mismatch"
    assert np.allclose(model.C, C_ref, atol=1e-10), "C matrix mismatch"
    print("test_ss_model_continuous_matrices_match_reference PASSED.")


def test_ss_model_discrete_matrices_populated():
    model = LinearStateSpaceModel(ballbeam_config)
    for attr in ('Ad', 'Bd', 'Cd', 'Dd'):
        assert hasattr(model, attr), f"Missing attribute {attr}"
        assert isinstance(getattr(model, attr), np.ndarray)
    print("test_ss_model_discrete_matrices_populated PASSED.")


def test_ss_model_discrete_matches_reference():
    """ZOH discretisation must agree with the hardcoded reference path."""
    model = LinearStateSpaceModel(ballbeam_config)
    A_ref, B_ref, C_ref, D_ref = _ref_ss_matrices()
    ss_ref = ct.StateSpace(A_ref, B_ref, C_ref, D_ref).sample(ballbeam_config.dt, 'zoh')

    assert np.allclose(model.Ad, ss_ref.A, atol=1e-10), "Ad mismatch"
    assert np.allclose(model.Bd, ss_ref.B, atol=1e-10), "Bd mismatch"
    assert np.allclose(model.Cd, ss_ref.C, atol=1e-10), "Cd mismatch"
    print("test_ss_model_discrete_matrices_match_reference PASSED.")


# ---------------------------------------------------------------------------
# 4. NonlinearBallBeamModel
# ---------------------------------------------------------------------------

def test_nonlinear_model_inherits_base():
    model = NonlinearBallBeamModel(ballbeam_config)
    assert isinstance(model, BaseNonlinearModel), \
        "NonlinearBallBeamModel must inherit BaseNonlinearModel"
    print("test_nonlinear_model_inherits_base PASSED.")


def test_nonlinear_model_has_C():
    model = NonlinearBallBeamModel(ballbeam_config)
    assert hasattr(model, 'C'), "NonlinearBallBeamModel must expose C"
    assert model.C.shape == (1, 2), f"Expected C shape (1,2), got {model.C.shape}"
    assert np.allclose(model.C, [[1.0, 0.0]]), "C must select ball position (first state)"
    print("test_nonlinear_model_has_C PASSED.")


def test_nonlinear_model_f_zero_input_zero_state():
    """At rest with zero control: derivative is zero."""
    model = NonlinearBallBeamModel(ballbeam_config)
    X = np.array([[0.0], [0.0]])
    dX = model.f(X, 0.0)
    assert np.allclose(dX, 0.0, atol=1e-12), \
        f"f(0, 0) should be zero, got {dX}"
    print("test_nonlinear_model_f_zero_input_zero_state PASSED.")


def test_nonlinear_model_f_velocity_propagates():
    """With non-zero velocity and zero input: position derivative equals velocity."""
    model = NonlinearBallBeamModel(ballbeam_config)
    v = 0.5
    X = np.array([[0.0], [v]])
    dX = model.f(X, 0.0)
    assert np.isclose(dX[0, 0], v, atol=1e-12), \
        f"ṙ should equal v={v}, got {dX[0,0]}"
    assert np.isclose(dX[1, 0], 0.0, atol=1e-12), \
        f"r̈ should be 0 with u=0, got {dX[1,0]}"
    print("test_nonlinear_model_f_velocity_propagates PASSED.")


def test_nonlinear_model_f_acceleration_physics():
    """Check r̈ matches the closed-form formula for a given control input."""
    model = NonlinearBallBeamModel(ballbeam_config)
    u = 1.0
    X = np.array([[0.0], [0.0]])
    dX = model.f(X, u)

    alpha = (model.d / model.L) * u
    r_ddot_expected = -model.m * model.g * np.sin(alpha) / (model.J / model.R**2 + model.m)

    assert np.isclose(dX[1, 0], r_ddot_expected, rtol=1e-10), \
        f"r̈ mismatch: got {dX[1,0]:.6f}, expected {r_ddot_expected:.6f}"
    print("test_nonlinear_model_f_acceleration_physics PASSED.")


def test_nonlinear_model_f_small_angle_matches_linear():
    """For small u, sin(α) ≈ α, so nonlinear r̈ ≈ H·u.

    H = -m·g·(d/L) / (J/R²+m) already absorbs the (d/L) factor, so the
    linearised acceleration is simply H·u, not H·(d/L)·u.
    """
    model = NonlinearBallBeamModel(ballbeam_config)
    u = 0.01
    X = np.array([[0.0], [0.0]])
    dX = model.f(X, u)

    r_ddot_linear = ballbeam_config.H * u
    assert np.isclose(dX[1, 0], r_ddot_linear, rtol=1e-4), \
        f"Small-angle nonlinear vs linear mismatch: {dX[1,0]:.6f} vs {r_ddot_linear:.6f}"
    print("test_nonlinear_model_f_small_angle_matches_linear PASSED.")


# ---------------------------------------------------------------------------
# 5. NonLinearHybridSim reads C from model
# ---------------------------------------------------------------------------

def test_nonlinear_hybridsim_C_from_model():
    model = NonlinearBallBeamModel(ballbeam_config)
    sim = NonLinearHybridSim(model, ballbeam_config)
    assert np.allclose(sim.C, model.C, atol=1e-12), \
        f"NonLinearHybridSim.C should equal model.C, got {sim.C}"
    print("test_nonlinear_hybridsim_C_from_model PASSED.")


# ---------------------------------------------------------------------------
# 6. Generic substitutability — mock plants
# ---------------------------------------------------------------------------

class _MockSSModel(BaseLinearStateSpaceModel):
    """Minimal first-order plant: ẋ = -x + u, y = x."""
    def __init__(self, dt):
        self.A = np.array([[-1.0]])
        self.B = np.array([[1.0]])
        self.C = np.array([[1.0]])
        self.D = np.array([[0.0]])
        ss = ct.StateSpace(self.A, self.B, self.C, self.D).sample(dt, 'zoh')
        self.Ad = np.array(ss.A)
        self.Bd = np.array(ss.B)
        self.Cd = np.array(ss.C)
        self.Dd = np.array(ss.D)


class _MockConfig:
    dt = 0.05
    T = 1.0


class _MockNonlinearModel(BaseNonlinearModel):
    """Minimal nonlinear plant: ẋ = -x³ + u, y = x."""
    def __init__(self):
        self.C = np.array([[1.0]])

    def f(self, X: np.ndarray, u: float) -> np.ndarray:
        return np.array([[-X[0, 0]**3 + u]])


def test_hybridsim_accepts_mock_ss_model():
    model = _MockSSModel(_MockConfig.dt)
    sim = HybridSim(model, _MockConfig())
    assert np.allclose(sim.A, model.A)
    assert np.allclose(sim.B, model.B)
    print("test_hybridsim_accepts_mock_ss_model PASSED.")


def test_nonlinear_hybridsim_accepts_mock_model():
    model = _MockNonlinearModel()
    sim = NonLinearHybridSim(model, _MockConfig())
    assert np.allclose(sim.C, model.C)
    # One RK4 step from zero state with zero input should stay at zero
    X = np.array([[0.0]])
    X_next = sim.rk4_step(X, np.array([[0.0]]))
    assert np.allclose(X_next, 0.0, atol=1e-12), \
        f"RK4 from zero with zero input should stay zero, got {X_next}"
    print("test_nonlinear_hybridsim_accepts_mock_model PASSED.")


def test_mock_nonlinear_model_cannot_be_abstract():
    """BaseNonlinearModel without f() must raise TypeError on instantiation."""
    class _IncompleteModel(BaseNonlinearModel):
        pass  # missing f()

    raised = False
    try:
        _IncompleteModel()
    except TypeError:
        raised = True
    assert raised, "Instantiating BaseNonlinearModel without f() should raise TypeError"
    print("test_mock_nonlinear_model_cannot_be_abstract PASSED.")


# ---------------------------------------------------------------------------

if __name__ == '__main__':
    test_config_has_required_tf_fields()
    test_config_has_required_ss_fields()

    test_tf_model_continuous_matches_reference()
    test_tf_model_discrete_poles_on_unit_circle()
    test_tf_model_num_den_arrays_extracted()
    test_tf_model_discrete_step_matches_reference()

    test_ss_model_inherits_base()
    test_ss_model_continuous_matrices_match_reference()
    test_ss_model_discrete_matrices_populated()
    test_ss_model_discrete_matches_reference()

    test_nonlinear_model_inherits_base()
    test_nonlinear_model_has_C()
    test_nonlinear_model_f_zero_input_zero_state()
    test_nonlinear_model_f_velocity_propagates()
    test_nonlinear_model_f_acceleration_physics()
    test_nonlinear_model_f_small_angle_matches_linear()

    test_nonlinear_hybridsim_C_from_model()

    test_hybridsim_accepts_mock_ss_model()
    test_nonlinear_hybridsim_accepts_mock_model()
    test_mock_nonlinear_model_cannot_be_abstract()

    print("\nAll model tests PASSED.")
