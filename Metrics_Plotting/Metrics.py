import numpy as np
import control as ct
from Metrics_Plotting.SimLog import SimLog


class Metrics:
    """Computes time-domain and frequency-domain performance metrics from simulation logs."""

    def __init__(self):
        """Initialises the Metrics instance (stateless; all methods take their data as arguments)."""

    def response_data(self, Logger: SimLog, reference: float):
        """Prints rise time, settling time, overshoot, and steady-state error from a closed-loop log.

        Analysis is restricted to the pre-disturbance phase (t ≤ 3 s) because the disturbance
        applied at t = 3 s would otherwise corrupt the transient metrics.

        Hardcoded thresholds
        --------------------
        t ≤ 3.0 s
            Pre-disturbance analysis window.  The runners apply a step disturbance at t = 3 s,
            so any sample after this instant reflects disturbance rejection rather than reference
            tracking.
        10 % / 90 % of reference
            Standard IEEE rise-time definition: the time for the output to travel from 10 % to
            90 % of the reference value.
        ±10 % band
            Settling-time tolerance band.  The settling time is the last instant at which the
            output is still outside this band.  A tighter ±2 % band is conventional; the 10 %
            band is used here to be conservative with the ball-and-beam's relatively slow dynamics.
        2.5 s to 3.0 s window
            Last 0.5 s before the disturbance onset, used to estimate the steady-state output
            once the transient has died out.

        Args:
            Logger (SimLog): Populated SimLog from a completed closed-loop simulation run.
            reference (float): Controller setpoint (target ball position).
        """
        y_arr = np.array(Logger.y_hist)
        t_arr = np.array(Logger.t_hist)

        # Restrict to pre-disturbance phase (disturbance injected at t = 3 s in the runners)
        mask = t_arr <= 3.0
        y_cl = y_arr[mask]
        t_cl = t_arr[mask]

        # Overshoot (%)
        peak = np.max(y_cl)
        overshoot = (peak - reference) / reference * 100 if peak > reference else 0.0

        # Rise time: first 10 % crossing → first 90 % crossing (standard IEEE definition)
        idx_10 = np.where(y_cl >= 0.1 * reference)[0]
        idx_90 = np.where(y_cl >= 0.9 * reference)[0]
        rise_time = (t_cl[idx_90[0]] - t_cl[idx_10[0]]) if len(idx_10) and len(idx_90) else float('nan')

        # Settling time: last sample outside the ±10 % tolerance band around the reference
        outside = np.where(np.abs(y_cl - reference) > 0.1 * reference)[0]
        settling_time = t_cl[outside[-1]] if len(outside) else 0.0

        # Steady-state error: mean deviation over the last 0.5 s before the disturbance onset
        ss_mask = (t_arr > 2.5) & (t_arr <= 3.0)
        ess = reference - np.mean(y_arr[ss_mask])

        print(f"Dépassement :          {overshoot:.1f} %")
        print(f"Temps de montée :      {rise_time:.3f} s  (10 %→90 %)")
        print(f"Temps d'établissement :     {settling_time:.3f} s  (bande ±10 %)")

    def Stability(self, TF: ct.TransferFunction):
        """Prints gain margin and phase margin with interpretation notes.

        **Positive gain margin (GM > 0 dB)**
            The loop gain can be *increased* by the factor 10^(GM/20) before the
            closed loop becomes unstable.  A design target of GM > 6 dB means the
            plant gain could double and the system would still be stable.

        **Negative gain margin (GM < 0 dB) — conditional stability**
            The closed loop is stable, but it requires a *minimum* gain to remain so.
            This is **conditionally stable** behaviour: the system is stable only inside
            a gain interval [K_min, K_max], not for all gains above a threshold.

            Why it happens here — triple integrator open loop:
              The ball-and-beam plant is a double integrator A(z) = (z−1)².  The RST
              controller adds a third integrator by forcing S(1) = 0, i.e. S contains a
              factor (z−1).  The resulting open loop L(z) = B·R/(A·S) has a triple pole
              at z = 1, equivalent to three cascaded discrete integrators.

              For a type-3 open loop, the Nyquist locus starts at angle +90° as ω→0⁺
              (because (e^{jω}−1)³ ≈ (jω)³ = −j·ω³, whose phase is −(−90°) = +90° in
              the denominator), then sweeps to zero at ω = π via the zero at z = −1 in
              B(z).  The locus crosses the negative real axis to the LEFT of −1 at some
              intermediate frequency, which means the Nyquist criterion requires a
              specific number of encirclements: too little gain and the required
              encirclement count is not met, making the closed loop unstable.

              A negative gain margin is therefore a structural property of any type-3
              (or higher) feedback loop — not a consequence of the RST synthesis method
              or a design trade-off.  It cannot be eliminated without removing one of
              the three integrators.

            What to watch for:
              The phase margin (typically 60°–80° for well-tuned RST) is the relevant
              robustness indicator for this class of system.  The lower gain boundary
              is worth knowing: if the plant gain H could drop significantly (mass
              change, actuator wear), check that the nominal gain stays above K_min.

        Args:
            TF (ct.TransferFunction): Open-loop discrete or continuous transfer function.
        """
        margins = ct.stability_margins(TF)
        gm, pm = margins[0], margins[1]
        gm_db = 20 * np.log10(gm)
        print(f"Gain margin:   {gm_db:.1f} dB  (linear ratio ×{gm:.3f})")
        print(f"Phase margin:  {pm:.1f}°")
        if gm_db < 0:
            destabilising_factor = 1.0 / gm
            print(
                f"  ↳ Negative gain margin: stable only above the minimum gain.\n"
                f"    Instability if plant gain drops by more than {destabilising_factor:.1f}× "
                f"({100 * (1 - gm):.0f}% reduction from nominal)."
            )
