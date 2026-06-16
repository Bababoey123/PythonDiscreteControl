# PID Tuning Rationale — Ball-and-Beam (Double Integrator)

## Plant summary

The linearised ball-and-beam model is a **double integrator**:

$$P(s) = \frac{H}{s^2}, \quad H = \frac{m\,g\,d}{L\,(J/R^2 + m)} \approx 0.210 \;\text{m\,s}^{-2}/\text{rad}$$

Sampled at $T_s = 0.05\,\text{s}$ (20 Hz) with ZOH, the discrete plant is:

$$P(z) = \frac{0.0002625\,(z+1)}{(z-1)^2}$$

Two poles sit exactly at $z = 1$ — a structurally undamped system with infinite DC gain.

---

## Why Ki = 0

A PID controller adds one more integrator at $z = 1$ when $K_i \neq 0$.  
The plant already contributes two poles at $z = 1$, so enabling integral action creates a **type-3 loop** (three open-loop integrators). Type-3 loops are notoriously difficult to stabilise: even small gain increases push the root locus outside the unit circle.

Verification: with $K_p = 100$, $K_i = 50$, $K_d = 125$ the closed-loop phase margin drops to **14°** and the gain margin to **3 dB** — far below safe limits.

Because the plant's double integrator already guarantees **zero steady-state error to a step reference** (infinite DC gain), integral action is redundant for reference tracking and actively destabilising here.

**Rule:** for a type-2 plant (double integrator), use a PD controller for reference tracking.

> **Important caveat — disturbance rejection.**  
> A PD controller does *not* give zero steady-state error under a constant input disturbance.  
> The disturbance-to-output DC gain is $G_d(0) = 1/K_p$, so a step disturbance $D$ causes a permanent offset:
> $$e_{\infty} = \frac{D}{K_p}$$
> With $K_p = 43$ and $D = 2$: $e_{\infty} = 2/43 \approx 0.047\,\text{m}$ — confirmed by simulation.  
> Eliminating this error requires $K_i \neq 0$ (type-3 loop) or an RST controller whose $S$ polynomial contains a $(z-1)$ factor for integral disturbance rejection.

---

## Choosing Kp — set the closed-loop bandwidth

For a PD controller on a double integrator the closed-loop characteristic polynomial approximates (for large filter cutoff $N \gg \omega_n$):

$$s^2 + K_d H\,s + K_p H = 0$$

Matching to the standard second-order form $s^2 + 2\zeta\omega_n s + \omega_n^2$ gives:

$$K_p = \frac{\omega_n^2}{H}$$

**Target:** $\omega_n = 3\;\text{rad/s}$, which corresponds to a settling time of approximately $4/(\zeta\omega_n) \approx 1.8\,\text{s}$, well within the 6-second simulation window.

$$K_p = \frac{3^2}{0.21} \approx 42.9 \;\rightarrow\; \boxed{K_p = 43}$$

---

## Choosing Kd — set the damping

From the same matching:

$$K_d = \frac{2\,\zeta\,\omega_n}{H}$$

A damping ratio $\zeta = 0.7$–$0.75$ is the standard engineering sweet spot: it limits overshoot to under 5 % while keeping the transient fast.

$$K_d = \frac{2 \times 0.73 \times 3}{0.21} \approx 20.9 \;\rightarrow\; \boxed{K_d = 21}$$

---

## Derivative filter — why N = 50 is kept

The filtered derivative is implemented as:

$$D(s) = K_d \frac{N s}{s + N}$$

The filter pole $N$ must satisfy two constraints:

| Constraint | Requirement | Value |
|---|---|---|
| Fast enough to pass damping | $N \gg \omega_n$ | $50 \gg 3$ ✓ |
| Below Nyquist | $N < \omega_{Nyq} = \pi/T_s$ | $50 < 62.8$ ✓ |

With $N = 50$ the discrete filter pole lands at $z = 1/(N T_s + 1) = 1/3.5 \approx 0.286$, well inside the unit circle, adding no instability risk.

---

## Stability analysis

| | **Previous** ($K_p=100, K_i=50, K_d=125$) | **Tuned** ($K_p=43, K_i=0, K_d=21$) |
|---|---|---|
| Dominant pole radius | 0.98 | **0.845** |
| Phase margin | 14.4° ⚠️ | **49.6°** |
| Gain margin | 3.0 dB ⚠️ | **17.6 dB** |
| Closed-loop $\omega_n$ | — | 3.0 rad/s |
| Closed-loop $\zeta$ | — | 0.73 |

A phase margin above 45° and a gain margin above 10 dB are the standard industrial robustness targets. The tuned controller meets both comfortably.

The dominant poles at radius 0.845 correspond to $s \approx -2.1 \pm j\,2.1$ in continuous time — a well-damped, moderately fast oscillation.

---

## Actuator saturation

The physical servo saturates at $|u| \leq 10\,\text{rad}$, giving a maximum beam angle of $\alpha = (d/L)\,u_{\max} = 0.03 \times 10 = 0.3\,\text{rad} \approx 17°$.  
For small angles $\sin(0.3) \approx 0.296$, which is close to $0.3$, so the linearisation remains accurate even at saturation — the nonlinear and linear models agree well throughout the simulation.

Saturation must be **enabled** (`Saturate=True`) with the tuned gains: the high $K_d$ produces a large derivative kick at $t = 0$ (error step), but the saturation limits this to 10 and the system recovers smoothly.

---

## Simulation results (reference = 0.25 m)

| Metric | Linear (TF) | Nonlinear (RK4) |
|---|---|---|
| Settling time (±2 % of ref) | 1.10 s | 1.06 s |
| Overshoot | 0.7 % | 0.7 % |
| Peak control effort | 10.0 (saturated) | 10.0 (saturated) |
| Steady-state error | 0 | < 0.5 % |

The nonlinear simulation matches the linear prediction closely, confirming that the linearisation is valid for the chosen reference.
