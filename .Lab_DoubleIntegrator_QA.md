# Lab Session: Discrete-Time Control of a Double Integrator
## Questions & Answers

---

## Part 1 — System Analysis

### Q1. Derive the discrete model from the continuous transfer function using ZOH ($T_s = 0.05$ s). Does it match the model in the notebooks?

**Answer:**

Starting from $G(s) = \frac{0.21}{s^2}$, applying a Zero-Order Hold discretization:

$$G(z) = (1 - z^{-1}) \mathcal{Z}\left\{\frac{G(s)}{s}\right\} = \frac{0.0002625(z + 1)}{z^2 - 2z + 1}$$

This matches the discrete model loaded in both notebooks exactly. The numerator coefficient $0.0002625 = 0.21 \cdot T_s^2 / 2$ comes from the ZOH integration of a double integrator.

---

### Q2. Plot the open-loop step response. Why is a double integrator difficult to control?

**Answer:**

The open-loop step response of a double integrator grows without bound (ramp-like divergence). This is because:

- Two poles at $s = 0$ (continuous) → two poles at $z = 1$ (discrete) mean the system has **no natural damping**.
- Any constant input produces an ever-increasing output — the system is **marginally unstable** in open loop.
- It cannot reject disturbances or track references without feedback.

This makes it a challenging but representative benchmark for controller design.

---

### Q3. What do the two poles at $z = 1$ mean in terms of stability?

**Answer:**

In the Z-domain, stability requires all poles to lie **strictly inside** the unit circle ($|z| < 1$). Poles at $z = 1$ lie exactly **on** the unit circle, meaning the system is:

- **Marginally stable** (not asymptotically stable) in open loop.
- The output neither decays nor grows for a zero input, but integrates any non-zero input indefinitely.

In practice, numerical errors or small disturbances will cause the output to drift. Feedback control is mandatory to stabilize the system and achieve a well-defined steady state.

---

## Part 2 — PID Tuning

### Q4. Run the PID notebook with $K_i = 0$. Why does the disturbance-to-output transfer function $G_d(z)$ have a non-zero DC gain ($\approx 0.004$)?

**Answer:**

The DC gain of $G_d(z)$ is evaluated at $z = 1$. When $K_i = 0$, the PID controller has **no integral action**, meaning it cannot produce the sustained corrective effort needed to fully cancel a constant (step) disturbance at steady state.

Mathematically, the closed-loop error due to a step disturbance converges to:

$$e_{ss} = \lim_{z \to 1} (z-1) \cdot G_d(z) \cdot \frac{1}{1 - z^{-1}} \neq 0$$

To achieve zero steady-state disturbance error, the loop transfer function must contain an integrator — which requires $K_i > 0$. Without it, the system settles at a position with a small but permanent offset.

---

### Q5. After enabling integral action ($K_i > 0$) and retuning, can you eliminate the disturbance residual while keeping reasonable stability margins?

**Answer:**

Yes. Adding $K_i > 0$ introduces an integrator in the controller, which forces $G_d(1) = 0$ at steady state. However, integral action **reduces phase margin**, so $K_p$ and $K_d$ must be adjusted:

- Increase $K_d$ to recover phase (derivative action adds phase lead).
- Reduce $K_p$ slightly if overshoot increases.
- A reasonable target: phase margin $> 30°$, gain margin $> 6$ dB.

The baseline design ($K_p=250$, $K_i=0$, $K_d=95$) already has only $17.2°$ of phase margin — adding $K_i$ without retuning will likely destabilize the loop. This illustrates why PID tuning for a double integrator is non-trivial and sensitive to gain choices.

---

### Q6. With only ~17° of phase margin in the baseline PID design, how sensitive is it to model uncertainty?

**Answer:**

Very sensitive. Phase margin is a measure of how much additional phase lag the loop can tolerate before going unstable. At $17.2°$:

- A small delay (e.g., one extra sampling period of $50$ ms adds ~$5$–$10°$ of phase lag at the crossover frequency) could destabilize the system.
- Parameter variations (e.g., a change in the ball mass shifting the gain from $0.21$ to $0.25$) reduce the gain margin further.
- In practice, a phase margin below $30°$ is generally considered risky for real hardware deployment.

This contrasts sharply with the RST design, which achieves $66°$ of phase margin through systematic synthesis.

---

## Part 3 — RST Polynomial Design

### Q7. When you change the desired closed-loop specs ($\omega_n$, $\zeta$), what happens to the control effort $u(t)$?

**Answer:**

- **Increasing $\omega_n$** (faster desired response) → larger RST gains → higher control effort $u(t)$, risk of hitting the $\pm 10$ V saturation limit.
- **Decreasing $\omega_n$** → smaller gains → lower control effort, but slower tracking.
- **Increasing $\zeta$** (more damping) → less overshoot but slower rise time; $u(t)$ peaks lower.
- **Decreasing $\zeta$** → faster but oscillatory response; $u(t)$ may oscillate and saturate.

There is a fundamental tradeoff: **faster response requires more control energy**. The baseline specs ($\omega_n = 2.2$ rad/s, $\zeta = 0.7$) represent a balanced choice that stays within the actuator limits.

---

### Q8. What happens to disturbance rejection speed when the observer pole is moved toward $z = 0$ (faster) vs. $z = 1$ (slower)?

**Answer:**

The observer polynomial $A_0(z) = z - p_0$ governs how fast the controller responds to disturbances:

- **$p_0 \to 0$ (fast observer):** The observer reacts quickly to disturbances, improving rejection speed. However, this amplifies measurement noise and increases control effort.
- **$p_0 \to 1$ (slow observer):** The observer reacts sluggishly; disturbance rejection is slow and the output dips more before recovering. This resembles a system with no active disturbance rejection.

The baseline value $p_0 = 0.9$ is a practical compromise: fast enough for good rejection, slow enough to filter noise. In the Bode plot of $G_d(z)$, moving $p_0$ toward 0 flattens the low-frequency response (better rejection) at the cost of higher high-frequency gain (more noise sensitivity).

---

### Q9. The RST design achieves a DC gain of $G_d(z) = 0$ exactly. Which design element ensures this, and why?

**Answer:**

The **integral constraint on the $S$ polynomial** ensures perfect disturbance rejection. In RST synthesis, an integrator is explicitly embedded in the sensitivity function by requiring $S(1) = 0$, i.e., the $S$ polynomial has a root at $z = 1$.

This means the controller contains an internal model of a step disturbance (the "internal model principle"): if the controller can generate a step signal, it can cancel any step disturbance at steady state.

In the notebook, this is enforced via the `S'` constraint in `computeRST.Compute_Denominator_Matching_RST()`. Unlike the PID case where integral action is optional ($K_i$ can be set to zero), the RST synthesis bakes it in by construction — disturbance rejection is guaranteed by the design algebra, not by the user remembering to set a gain.

---

## Part 4 — Comparison and Discussion

### Q10. The RST has better phase margin but slower transient response than the baseline PID. Is this a fundamental tradeoff or a tuning choice?

**Answer:**

It is primarily a **tuning choice**, not a fundamental tradeoff — but one that reveals a real tension in control design.

The RST baseline was tuned for robustness ($\omega_n = 2.2$ rad/s, $\zeta = 0.7$), giving a slow, well-damped response with large stability margins. The PID baseline was tuned for speed ($K_d = 95$ pushes bandwidth higher), accepting lower margins in exchange.

You could tune the RST for a faster response (increase $\omega_n$) and reduce its phase margin, or tune the PID for more robustness and slow it down. However, in practice:

- The RST synthesis gives you **explicit control over the tradeoff** via $\omega_n$ and $\zeta$.
- The PID requires trial-and-error and the relationship between gains and margins is less transparent.

So both approaches can reach similar operating points, but RST makes the design intent explicit and repeatable.

---

### Q11. Which controller would you deploy on real hardware and why?

**Answer:**

**RST is preferable for real hardware deployment**, for several reasons:

1. **Robustness:** $66°$ phase margin vs. $17.2°$ — the RST design tolerates model errors, delays, and parameter drift far better.
2. **Guaranteed disturbance rejection:** Perfect step disturbance rejection ($G_d$ DC gain $= 0$) is built into the design, not reliant on a gain setting.
3. **Systematic design:** If the physical parameters change (e.g., different ball mass), rerunning the RST synthesis with updated specs gives a new valid design immediately. Retuning a PID requires manual iteration.
4. **Predictable behavior:** The closed-loop response matches the specified second-order model, making it easier to validate against requirements.

The PID is acceptable for rapid prototyping or when the system is well-known and manually tuned gains can be validated extensively. But for a safety-relevant or precision application, the RST's formal synthesis approach is more trustworthy.

---

### Q12. What would change if the sampling period were increased to $T_s = 100$ ms?

**Answer:**

Several things change:

1. **Discrete model:** The numerator coefficient scales as $T_s^2 / 2$, so the gain doubles. The poles stay at $z = 1$ but the model dynamics shift.

2. **Phase lag:** A slower sampling rate adds more phase lag at a given frequency. For a crossover frequency near $2$ rad/s, an extra $50$ ms of delay adds approximately $\Delta\phi \approx -\omega \cdot T_s \approx -5.7°$ — significant given the PID's already-thin margin.

3. **Disturbance rejection:** The controller can only react once per sample. A disturbance injected between samples goes undetected for up to $100$ ms, worsening the disturbance dip.

4. **RST redesign required:** The desired poles are discretized via $z = e^{sT_s}$, so new RST polynomials must be computed. The design procedure is the same, but the resulting coefficients differ.

5. **Control effort:** Larger $T_s$ means the controller acts less frequently, potentially requiring larger actuator steps to compensate — higher risk of saturation.

In general, halving the sampling rate degrades performance and robustness, particularly for a system as sensitive as the double integrator near its marginally stable poles.
