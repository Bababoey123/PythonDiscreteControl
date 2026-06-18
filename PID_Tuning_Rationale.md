# PID Tuning Rationale — Ball-and-Beam (Backward Euler, dt = 0.05 s)

## Final Gains

| Parameter | Value |
|-----------|-------|
| Kp | 70 |
| Ki | 2 |
| Kd | 30 |
| Filter coefficient N | 50 |

---

## Constraints

| Constraint | Target | Result |
|------------|--------|--------|
| Gain margin (GM) | ≥ 10 dB | **14.2 dB** ✓ |
| Phase margin (PM) | ≥ 45° | **45.7°** ✓ |
| Overshoot | < 5 % | **1.98 %** ✓ |
| Integral action | Ki ≠ 0 | **Ki = 2** ✓ |
| Rise time | — | 0.750 s |
| Settling time (2 %) | — | 1.200 s |

---

## Plant

The ball-and-beam is linearized as a double integrator:

```
G(s) = H / s²,   H ≈ 0.2100 m·s⁻²·rad⁻¹
```

ZOH discretization at dt = 0.05 s:

```
G_d(z) = H·dt²/2 · (z + 1) / (z − 1)²
       = 2.625×10⁻⁴ · (z + 1) / (z − 1)²
```

This discretization is **exact** (not approximate) because A² = 0 for the double integrator, so the matrix exponential terminates at the first-order term.

---

## PID Structure (Backward Euler)

Each term is discretized via the bilinear mapping s → (z − 1)/(dt·z):

```
P(z) = Kp
I(z) = Ki · dt · z / (z − 1)
D(z) = Kd · N · (z − 1) / ((N·dt + 1)·z − 1)
```

With N = 50, dt = 0.05 s the derivative filter pole is at z = 1/(N·dt + 1) = 0.286 — well inside the unit circle, providing high-frequency roll-off above ≈ 14 rad/s.

---

## Why Integral Action Is Required

The plant is Type 2 (double integrator), so a proportional-derivative controller already gives zero steady-state error for step references. However, **Ki = 0 leaves a permanent offset after a step input disturbance**:

```
Δy_ss = D / Kp   (for Ki = 0, D = disturbance magnitude)
```

With D = 2 and Kp = 70 this would be 0.029 m — unacceptable for precise positioning.

Adding Ki ≠ 0 promotes the open loop to **Type 3**, making S_PID(1) = 0:

```
F_p^in(z=1) = S_PID(1) / R_PID(1) = 0 / R_PID(1) = 0
```

Exact disturbance rejection is achieved, identical in steady state to the RST controller with an integrator in S.

---

## Conditional Stability (Type-3 Loop)

A Type-3 open loop is **conditionally stable**: the Nyquist locus crosses the negative real axis at two frequencies, creating both a lower and an upper gain margin.

For the final design, the three phase crossings of L(z) = C_PID(z)·G_d(z) are:

| Crossing | Frequency (rad/s) | \|L\| | GM at crossing |
|----------|------------------|-------|----------------|
| Lower conditional | 0.27 | 208.7 | −46.4 dB |
| Upper (conventional) | 24.0 | 0.194 | +14.2 dB |
| Second upper | 62.7 | 0.0003 | +71.9 dB |

`python-control` reports the conventional upper GM of **+14.2 dB**. The lower conditional crossing at |L| = 208.7 means the loop requires the plant gain to remain above nominal/208.7 to stay stable — an irrelevant constraint for the physical system.

The negative conditional GM is a **structural property** of any Type-3 feedback loop, not a design flaw. It cannot be eliminated without removing one integrator.

---

## Tuning Methodology

An initial grid search maximized closed-loop bandwidth (gain crossover frequency ωc). This produced high-Kd/low-Kp designs with ωc ≈ 10–12 rad/s that appeared optimal in the frequency domain. However, with a 3 s simulation window these designs showed settling times clipped at T = 3 s — the wide-bandwidth designs were exciting lightly damped closed-loop poles, producing slow settling despite fast rise times.

The simulation was extended to T = 6 s and the search metric was changed to minimize **RT + ST(2%)** (rise time + 2 % settling time) directly from the time-domain response. The final grid was:

```
Kp ∈ [40, 80, step 5],  Ki ∈ [1, 2, 3, 5],  Kd ∈ [15, 30, step 5]
```

The optimal point Kp = 70, Ki = 2, Kd = 30 was selected as the design that minimized RT + ST(2 %) subject to GM ≥ 10 dB, PM ≥ 45°, and OS < 5 %.

---

## Comparison with Previous Tuning (Kp = 43, Ki = 0, Kd = 21)

| Metric | Previous | New |
|--------|----------|-----|
| Kp / Ki / Kd | 43 / 0 / 21 | 70 / 2 / 30 |
| Gain margin | 17.6 dB | 14.2 dB |
| Phase margin | 49.6° | 45.7° |
| Overshoot | ~0 % | 1.98 % |
| Rise time | 0.850 s | 0.750 s |
| Settling time (2 %) | 1.400 s | 1.200 s |
| Disturbance rejection | D/Kp ≈ 0.047 m offset | Exact (0 m offset) |

The new design accepts a slightly lower but still comfortable gain margin (+4.2 dB above target) and a small overshoot (~2 %) in exchange for faster dynamics and perfect disturbance rejection.
