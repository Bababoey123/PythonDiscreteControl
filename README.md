# PythonDiscreteControl

A lightweight Python library for simulating, designing, and validating discrete and hybrid control systems. The library makes the gap between theoretical discrete models and physical sampled-data systems explicit and measurable.
Full documentation can be accessed [here](https://bababoey123.github.io/PythonDiscreteControl/)
---

## Table of Contents

- [Architecture](#architecture)
- [Notebooks](#notebooks)
- [Plant Model](#plant-model)
  - [Generic model interface](#generic-model-interface)
  - [Ball-and-beam (reference plant)](#ball-and-beam-reference-plant)
  - [Adding a new plant](#adding-a-new-plant)
- [Simulation Layer](#simulation-layer)
- [Controllers](#controllers)
- [RST Synthesis](#rst-synthesis)
- [Utility Functions](#utility-functions)
- [Simulation Runners](#simulation-runners)
- [Logging and Plotting](#logging-and-plotting)
- [Sampled-Data Behaviour](#sampled-data-behaviour)
- [Dependencies](#dependencies)

---

## Architecture

```
PythonDiscreteControl_Dev/
├── Models/
│   ├── base.py                    # Abstract base classes (BaseNonlinearModel, BaseLinearStateSpaceModel)
│   ├── BallBeam/
│   │   ├── ballbeam_config.py     # Physical parameters, timing, and generic model descriptors
│   │   ├── TransferFunctions.py   # Continuous and ZOH-discrete plant TF (reads from config)
│   │   ├── StateSpace.py          # Linearised continuous + discrete state-space (reads from config)
│   │   └── NonlinearDynamics.py   # Nonlinear ODE model (full sin(α) term)
│   └── tests/
│       └── test_models.py         # Unit tests for the Models layer
├── Simulation/
│   ├── simulation.py              # TFSimulator, HybridSim, NonLinearHybridSim
│   └── runners.py                 # High-level simulation entry points
├── Control/
│   ├── DiscretePID.py             # Backward-Euler PID controller
│   └── RSTController.py           # Two-degree-of-freedom RST controller
├── Utils/
│   ├── computeRST.py              # RST polynomial synthesis (Diophantine)
│   └── utils.py                   # Pole placement helpers, CSV export
└── Metrics_Plotting/
    ├── SimLog.py                  # Simulation data recorder
    ├── Plotting.py                # Time-domain plots
    └── Metrics.py                 # Response metrics (rise time, settling time, margins)
```

Each layer has a strict interface contract: controllers must expose `step(y)` and `setReference(r)`; simulators accept any model that satisfies the base-class interface defined in `Models/base.py`.

---

## Notebooks

Three Jupyter notebooks demonstrate the full design workflow on the ball-and-beam plant:

| Notebook | Content |
|---|---|
| `DoubleIntégrateurAnalyse.ipynb` | Open-loop analysis — step and impulse responses, ZOH discretisation, continuous vs discrete comparison |
| `CommandeDoubleIntégrateur_PID.ipynb` | Discrete PID design — gain tuning, stability margins, disturbance rejection |
| `CommandeDoubleIntégrateur_RST.ipynb` | RST polynomial design — Diophantine synthesis, observer polynomial, linear and nonlinear closed-loop simulation |

Run cells in order from top to bottom. Use **Run → Run All Cells** for a clean execution from a fresh kernel.

---

## Plant Model

### Generic model interface

The Models layer is built around two abstract base classes in `Models/base.py`:

- **`BaseNonlinearModel`** — enforces a `f(X, u)` method (state derivative) and a `C` output matrix. Python raises `TypeError` if a subclass forgets to implement `f`.
- **`BaseLinearStateSpaceModel`** — type marker that subclasses populate with `A, B, C, D` (continuous) and `Ad, Bd, Cd, Dd` (ZOH-discrete).

The simulators `HybridSim` and `NonLinearHybridSim` depend only on these base classes, so any new plant can be dropped in without touching the simulation layer. See *Adding a new plant* below.

### Ball-and-beam (reference plant)

The reference plant is the **ball-and-beam system**, linearised around the equilibrium as a **double integrator**:

```
G(s) = H / s²
```

where `H = m·g·d / (L · (J/R² + m))` is computed from the physical parameters in `ballbeam_config.py`.

| Parameter | Symbol | Value |
|---|---|---|
| Ball mass | m | 0.111 kg |
| Ball radius | R | 0.015 m |
| Ball moment of inertia | J | 9.99×10⁻⁶ kg·m² |
| Gravitational acceleration | g | −9.8 m/s² |
| Lever arm | d | 0.03 m |
| Beam length | L | 1.0 m |
| Linearised gain | H | ≈ 0.210 |
| Default sampling period | dt | 0.02 s (50 Hz) |

`ballbeam_config.py` also exposes generic model descriptors consumed by `TransferFunctionModel` and `LinearStateSpaceModel`:

```python
num_cont = [H]           # continuous TF numerator
den_cont = [1, 0, 0]    # continuous TF denominator
A_mat    = [[0, 1], [0, 0]]
B_mat    = [[0], [H]]
C_mat    = [[1, 0]]
D_mat    = [[0]]
```

The ZOH-discrete plant is:

```
G_d(z) = H·dt²/2 · (z + 1) / (z − 1)²
```

This discretisation is **exact** for the double integrator because A² = 0 (the matrix exponential is a finite polynomial in A).

Both representations are available:

```python
from Models.BallBeam.TransferFunctions import TransferFunctionModel
from Models.BallBeam.StateSpace import LinearStateSpaceModel
from Models.BallBeam import ballbeam_config

model = TransferFunctionModel(ballbeam_config)  # model.Tf_cont, model.Tf_dis
ss    = LinearStateSpaceModel(ballbeam_config)  # ss.A, ss.B, ss.C, ss.Ad, ss.Bd
```

The nonlinear model uses the full `sin(α)` term instead of the small-angle approximation:

```python
from Models.BallBeam.NonlinearDynamics import NonlinearBallBeamModel

nl = NonlinearBallBeamModel(ballbeam_config)   # nl.f(X, u) — ODE right-hand side
```

### Adding a new plant

1. Create `Models/MyPlant/myplant_config.py` with `num_cont`, `den_cont`, `A_mat`, `B_mat`, `C_mat`, `D_mat`, `dt`, `T`.
2. Subclass `BaseNonlinearModel` and implement `f(X, u)` and set `self.C`.
3. Pass the config to `TransferFunctionModel` and `LinearStateSpaceModel` — they work as-is.
4. Pass the model to `HybridSim` or `NonLinearHybridSim` — no changes required.

```python
from Models.base import BaseNonlinearModel

class MyPlantModel(BaseNonlinearModel):
    def __init__(self, config):
        self.C = np.array(config.C_mat)
        # store physical params ...

    def f(self, X, u):
        # return state derivative
        ...
```

---

## Simulation Layer

### `TFSimulator`

Implements any discrete transfer function as a finite difference equation:

```
a₀·y[k] = b₀·u[k] + b₁·u[k−1] + … − a₁·y[k−1] − a₂·y[k−2] − …
```

```python
from Simulation.simulation import TFSimulator

sim = TFSimulator(model.Tf_dis, X_0=0)
y = sim.step(u)   # advances one sample
sim.reset(0)
```

The coefficients are read directly from the `ct.TransferFunction` numerator/denominator arrays, so any python-control TF — plant, controller, or filter — can be wrapped.

### `HybridSim`

Couples a **continuous linear state-space plant** (RK4 at `dt_plant = 1 ms`) with a **discrete controller** (ZOH at `dt`). This emulates a real embedded control loop where the microcontroller samples the sensor and holds the actuator command for one period.

```python
from Simulation.simulation import HybridSim

hybrid = HybridSim(ss, ballbeam_config)
X_next = hybrid.rk4_step(X, u_k)   # advance one integration step
```

### `NonLinearHybridSim`

Identical structure to `HybridSim` but uses the **nonlinear ODE** `f(X, u)` from any `BaseNonlinearModel` subclass instead of linear state-space matrices. The output matrix `C` is read directly from the model.

```python
from Simulation.simulation import NonLinearHybridSim
from Models.BallBeam.NonlinearDynamics import NonlinearBallBeamModel

nl_model = NonlinearBallBeamModel(ballbeam_config)
nl_sim   = NonLinearHybridSim(nl_model, ballbeam_config)
X_next   = nl_sim.rk4_step(X, u_k)
```

Both hybrid simulators integrate at `dt_plant = 1 ms`. The ratio of control period to integration step depends on the chosen `dt` (e.g. 20× for `dt = 20 ms`, 50× for `dt = 50 ms`).

---

## Controllers

All controllers share the same interface:

```python
controller.setReference(r)   # set the reference before the loop
u = controller.step(y)       # called once per control period
```

### `DiscretePID`

Backward-Euler discretisation of `C(s) = Kp + Ki/s + Kd·s`.

**Unfiltered form** (velocity PID, `u[k] − u[k−1] = b₀·e[k] + b₁·e[k−1] + b₂·e[k−2]`):

```
b₀ = Kp + Ki·dt + Kd/dt
b₁ = −Kp − 2·Kd/dt
b₂ = Kd/dt
```

**Filtered derivative** (`text_option="filtered"`): the D term is replaced by a first-order filter `Kd·N·(z−1) / ((1+N·dt)·z − 1)` with `N = 50`.

```python
from Control.DiscretePID import DiscretePID

pid = DiscretePID(kp=40, ki=60, kd=5, dt=ballbeam_config.dt)
```

The controller also exposes its equivalent RST polynomials (`pid.R`, `pid.S`, `pid.T`) for direct use with `RSTController`.

> **Note on gains**: with a 20 ms sampling period, `Kd/dt` can become very large (e.g. `Kd=50` → `Kd/dt = 2500`). Large derivative gains cause significant inter-sample oscillations in the continuous plant that the discrete model does not capture. Keep `Kd` small relative to `dt`.

### `RSTController`

Two-degree-of-freedom polynomial controller implementing:

```
S(z)·u[k] = T(z)·r[k] − R(z)·y[k]
```

The `1/S` block is a recursive `TFSimulator`; `T` and `R` are applied as direct FIR dot products on the reference and output histories respectively.

```python
from Control.RSTController import RSTController

rst = RSTController(R_tf, S_tf, T_tf)
```

R, S, T can be obtained from the PID equivalent representation or from the RST synthesis functions below.

---

## RST Synthesis

`Utils/computeRST.py` implements polynomial RST synthesis by solving the **Diophantine equation**:

```
A(z)·S(z) + B(z)·R(z) = A_m(z)·A0(z)
```

where `A`, `B` are the plant denominator and numerator, `A_m` contains the **dominant closed-loop poles**, and `A0` is the **observer polynomial** (faster auxiliary poles).

### `Compute_Denominator_Matching_RST`

Pole placement by specifying the desired dominant denominator and an optional observer polynomial.

```python
from Utils.computeRST import Compute_Denominator_Matching_RST

A_m = [1, -1.2, 0.4]                # desired dominant poles
A0  = [1, -0.5]                     # observer pole at z = 0.5 (faster than A_m)

S, R, T, H_cl = Compute_Denominator_Matching_RST(
    A_m,
    plant_discrete_tf = model.Tf_dis,
    Integrator = True,               # forces (z-1) factor in S
    A0 = A0                          # optional; defaults to [1] (no observer pole)
)
```

**`Integrator=True`** (default): forces `S = (z−1)·S̃`, guaranteeing zero steady-state error for step references and exact rejection of constant disturbances (`S(1) = 0`).

**`A0`** (observer polynomial): its roots become additional closed-loop poles, placed faster than the dominant ones. `T = t₀·A0` ensures A0 cancels from the reference-to-output transfer function:

```
H_ry = B·T / (A·S + B·R) = B·t₀·A0 / (A_m·A0) = t₀·B / A_m
```

The closed-loop response therefore follows the dominant model `A_m` only, independent of `A0`.

**Solve strategy**: when the full characteristic polynomial `A_m·A0` fits in the Diophantine system, the equation is solved **directly** against `A_m·A0`. When it does not fit, the function falls back to a **two-step Landau factorisation** (solve against `A_m`, then apply `S = A0·S'`, `R = A0·R'`). Both paths enforce `S[0] > 0`, which is required for the `1/S` filter to drive the plant in the correct direction.

**Degree constraint**: `deg(A_m)` must not exceed the system capacity (number of Diophantine equations). `deg(A_m) + deg(A0)` can exceed it when the Landau fallback is used.

### `Compute_Desired_RST`

Model-following synthesis: specify the full desired closed-loop transfer function (numerator **and** denominator).

```python
from Utils.computeRST import Compute_Desired_RST

Desired_TF = utils.Place_real_radius(model.Tf_dis, pole_radius=0.9, steady_gain=1.0)
S, R, T, H_cl = Compute_Desired_RST(
    Desired_TF,
    plant_discrete_tf = model.Tf_dis,
    Integrator = True,
    A0 = None
)
```

T is computed by exact polynomial division `B_cl / B`, enforcing the desired closed-loop numerator. The Diophantine solves for the denominator matching `A_cl·A0`.

### Theory

Both functions use a **Toeplitz convolution matrix** (Sylvester matrix) to set up the linear system. The least-squares solver `np.linalg.lstsq` is used (exact solution when the system is square; least-squares fallback otherwise).

Stability of the desired poles and A0 is checked automatically:
- `Compute_Denominator_Matching_RST` rejects poles strictly outside the unit circle; warns on poles on the unit circle.
- `Compute_Desired_RST` rejects poles on or outside the unit circle.

---

## Utility Functions

### `Place_real_radius`

Builds a desired closed-loop TF with **all real poles** at a given radius, compatible with `Compute_Desired_RST`.

```python
from Utils.utils import Place_real_radius

Desired_TF = Place_real_radius(model.Tf_dis, pole_radius=0.85, steady_gain=1.0)
```

The denominator has degree `deg(A) + deg(B)` and all poles at `pole_radius`. The numerator is scaled to achieve `steady_gain` at DC.

### `poles_to_denominator`

Converts a list of desired z-domain poles (real or complex) to a monic denominator polynomial. Complex poles must come in conjugate pairs for a real-coefficient result.

```python
from Utils.utils import poles_to_denominator

j = complex(0, 1)
poles = [0.8, 0.82 + 0.2j, 0.82 - 0.2j]
A_cl = poles_to_denominator(poles)        # real-coefficient monic array
```

Raises `ValueError` if any pole is on or outside the unit circle (`check_stability=True`).

### `as_csv`

Exports a `SimLog` to a CSV file with columns `time, output, input`.

```python
from Utils.utils import as_csv

as_csv('experiment_01', logger)   # writes experiment_01.csv
```

---

## Simulation Runners

`Simulation/runners.py` provides six ready-to-use simulation functions. All accept a `SimLog` instance and return it populated with data.

### Open-loop discrete

```python
from Simulation.runners import run_discrete_step_response, run_discrete_impulse_response

step_log    = run_discrete_step_response(plant_sim, ballbeam_config, y_0=0.0, logger=SimLog())
impulse_log = run_discrete_impulse_response(plant_sim, ballbeam_config, y_0=0.0, logger=SimLog())
```

The impulse magnitude is `1/dt` to approximate a continuous Dirac delta with unit area.

### Closed-loop discrete

```python
from Simulation.runners import run_discrete_control

log = run_discrete_control(
    plant_sim, controller, ballbeam_config,
    r=1.0, y_0=0.0, logger=SimLog(),
    Disturb=False,   # True: adds step disturbance of 2.0 after t = 3 s
    Saturate=True    # True: clips u to [−10, +10]
)
```

### Open-loop continuous (hybrid plant)

```python
from Simulation.runners import run_continuous_step_response, run_continuous_impulse_response

step_log    = run_continuous_step_response(hybrid, X_0, SimLog())
impulse_log = run_continuous_impulse_response(hybrid, X_0, SimLog())
```

Both use RK4 at `dt_plant = 1 ms`. The impulse magnitude is `1/dt_plant`.

### Closed-loop hybrid

Works with both `HybridSim` (linear plant) and `NonLinearHybridSim` (nonlinear plant).

```python
from Simulation.runners import run_continuous_control_loop

log = run_continuous_control_loop(
    hybrid_or_nl_sim, controller, r=1.0, X_0=np.array([[0],[0]]),
    Logger=SimLog(),
    Disturb=False,   # True: adds step disturbance of 2.0 after t = 3 s
    Saturate=True    # True: clips u to [−10, +10]
)
```

The controller updates at `dt` (ZOH); the plant integrates at `dt_plant`. Plant output is sampled **once at the start of each control period**, exactly matching the ZOH sampled-data model.

---

## Logging and Plotting

### `SimLog`

Records `(t, y, u)` at every simulation step:

```python
from Metrics_Plotting.SimLog import SimLog

log = SimLog()
log.log(t, np.array([y]), np.array([u]))

# Access data
log.t_hist   # list of timestamps
log.y_hist   # list of scalar output values
log.u_hist   # list of scalar input values
```

### `Plotting`

```python
from Metrics_Plotting.Plotting import Plotting

plot = Plotting()
plot.plotAll(log, title="Step response")   # two figures: y(t) and u(t)
```

### `Metrics`

```python
from Metrics_Plotting.Metrics import Metrics

m = Metrics()
m.response_data(logger, reference)   # prints overshoot, rise time, settling time
m.Stability(open_loop_tf)            # prints gain margin and phase margin
```

`response_data` analyses only the pre-disturbance phase (`t ≤ 3 s`). Rise time uses the standard 10 %→90 % IEEE definition; settling time uses a ±10 % tolerance band.

---

## Sampled-Data Behaviour

A key design goal of this library is to make the difference between a **discrete model** and a **hybrid simulation** observable.

**Discrete model** (`ct.step_response` or `run_discrete_control`): operates entirely at the sampling rate. Only the output at `t = k·dt` is known. Connecting samples with `plt.step()` produces a staircase that hides what happens between samples.

**Hybrid simulation** (`run_continuous_control_loop`): the controller still updates every `dt`, but the continuous plant is integrated at `dt_plant = 1 ms`. A sensor logging at 1 ms would see the full inter-sample trajectory.

For the double integrator, the output between two samples follows a **parabolic arc** (constant ZOH input → constant acceleration). With aggressive derivative gains these arcs can be much larger than the reference value, even when the discrete closed-loop is stable.

```
At sampling instant k: both models agree exactly.
Between samples:       only the hybrid simulation captures the true motion.
```

This is why plotting a hybrid log and a `ct.step_response` on the same axes looks different: they are computing different things, not revealing a bug.

To compare them fairly, subsample the hybrid logger at sampling instants:

```python
indices = [i for i, t in enumerate(log.t_hist)
           if abs(t % ballbeam_config.dt) < 5e-4]
t_s = [log.t_hist[i] for i in indices]
y_s = [log.y_hist[i] for i in indices]
```

---

## Dependencies

```text
numpy
scipy
control (python-control)
scikit-learn
matplotlib
```

Install with:

```bash
pip install numpy scipy control scikit-learn matplotlib
```
