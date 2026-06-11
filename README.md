# PythonDiscreteControl

A lightweight Python library for simulating, designing, and validating discrete and hybrid control systems. The library makes the gap between theoretical discrete models and physical sampled-data systems explicit and measurable.

---

## Table of Contents

- [Architecture](#architecture)
- [Plant Model](#plant-model)
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
│   └── BallBeam/
│       ├── ballbeam_config.py      # Physical parameters and timing constants
│       ├── TransferFunctions.py    # Continuous and ZOH-discrete plant TF
│       └── StateSpace.py          # Linearised continuous + discrete state-space
├── Simulation/
│   ├── simulation.py              # TFSimulator, HybridSim
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
    └── Metrics.py                 # Response metrics (placeholder)
```

Each layer has a strict interface contract: controllers must expose `step(y)` and `setReference(r)`; simulators accept any object satisfying this interface.

---

## Plant Model

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
| Sampling period | dt | 0.02 s (50 Hz) |

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

Couples a **continuous state-space plant** (RK4 at `dt_plant = 1 ms`) with a **discrete controller** (ZOH at `dt`). This emulates a real embedded control loop where the microcontroller samples the sensor and holds the actuator command for one period.

```python
from Simulation.simulation import HybridSim

hybrid = HybridSim(ss, ballbeam_config)
X_next = hybrid.rk4_step(X, u_k)   # advance one integration step
```

The inner integration step (`dt_plant = 1 ms`) is 20× finer than the control period (`dt = 20 ms`) by default.

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
A(z)·S(z) + B(z)·R(z) = A_cl(z)·A0(z)
```

where `A`, `B` are the plant denominator and numerator, `A_cl` contains the **dominant closed-loop poles**, and `A0` is the **observer polynomial**.

### `Compute_Denominator_Matching_RST`

Pole placement by specifying the desired closed-loop denominator directly.

```python
from Utils.computeRST import Compute_Denominator_Matching_RST

A_cl = [1, -1.2, 0.4, -0.064]          # desired dominant poles
S, R, T, H_cl = Compute_Denominator_Matching_RST(
    A_cl,
    plant_discrete_tf = model.Tf_dis,
    Integrator = True,                  # forces (z-1) factor in S
    A0 = None                           # optional observer polynomial
)
```

**`Integrator=True`** (default): forces `S = (z−1)·S̃`, guaranteeing zero steady-state error for step references. The Diophantine is then exactly determined (rows = unknowns).

**`Integrator=False`**: standard form, system is underdetermined by one equation (zero-padded target).

**`A0`** (observer polynomial): its roots become **additional closed-loop poles**, placed faster than the dominant ones. The total characteristic polynomial is `A_cl·A0`. Degree constraint: `deg(A_cl) + deg(A0) ≤ deg(A) + deg(B)`.

```python
# Example: add two observer poles at z = 0.5
A0 = [1, -0.5]   # (z - 0.5)
S, R, T, H_cl = Compute_Denominator_Matching_RST(A_cl, model.Tf_dis, A0=A0)
```

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

Both functions use a **Toeplitz convolution matrix** to set up the linear system. The least-squares solver `np.linalg.lstsq` is used (exact solution when the system is square; least-squares fallback otherwise).

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
from Simulation.runners import run_continuous_step_response, run_continuous_impulse_respone

step_log    = run_continuous_step_response(hybrid, X_0, SimLog())
impulse_log = run_continuous_impulse_respone(hybrid, X_0, SimLog())
```

Both use RK4 at `dt_plant = 1 ms`. The impulse magnitude is `1/dt_plant`.

### Closed-loop hybrid

```python
from Simulation.runners import run_continuous_control_loop

log = run_continuous_control_loop(
    hybrid, controller, r=1.0, X_0=np.array([[0],[0]]),
    Logger=SimLog(),
    Disturb=False,   # True: adds step disturbance of 6.0 after t = 3 s
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

```
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
