import numpy as np
import control as ct
from Metrics_Plotting.SimLog import SimLog

class Metrics:
    """Placeholder for computing time-response metrics from a SimLog.

    Not yet implemented.
    """
    def __init__(self):
        """Initialises the Metrics instance (no state required)."""
        return

    def response_data(self, Logger:SimLog,reference:float):
        """Extracts response metrics (rise time, settling time, overshoot) from a log.

        Args:
            Logger (SimLog): Populated SimLog instance from a completed simulation run.
            reference (float): the reference of the controller

        Returns:
            dict: Dictionary of response metrics (not yet implemented; returns None).
        """
        y_arr = np.array(Logger.y_hist)
        t_arr = np.array(Logger.t_hist)

        # Restrict analysis to the tracking phase (before disturbance onset)
        mask = t_arr <= 3.0
        y_cl = y_arr[mask]
        t_cl = t_arr[mask]

        # Overshoot (%)
        peak = np.max(y_cl)
        overshoot = (peak - reference) / reference * 100 if peak > reference else 0.0

        # Rise time: first crossing of 10 % to first crossing of 90 %
        idx_10 = np.where(y_cl >= 0.1 * reference)[0]
        idx_90 = np.where(y_cl >= 0.9 * reference)[0]
        rise_time = (t_cl[idx_90[0]] - t_cl[idx_10[0]]) if len(idx_10) and len(idx_90) else float('nan')

        # Settling time: last sample outside the ±2 % band
        outside = np.where(np.abs(y_cl - reference) > 0.1 * reference)[0]
        settling_time = t_cl[outside[-1]] if len(outside) else 0.0

        # Steady-state error (mean over last 0.5 s before disturbance)
        ss_mask = (t_arr > 2.5) & (t_arr <= 3.0)
        ess = reference - np.mean(y_arr[ss_mask])

        print(f"Dépassement :          {overshoot:.1f} %")
        print(f"Temps de montée :      {rise_time:.3f} s  (10 %→90 %)")
        print(f"Temps d'établissement :     {settling_time:.3f} s  (bande ±10 %)")
        return
    def Stability(self,TF:ct.TransferFunction):
        margins = ct.stability_margins(TF)
        gm, pm = margins[0], margins[1]
        print(f"Gain margin:   {20*np.log10(gm):.1f} dB  (×{gm:.2f})")
        print(f"Phase margin:  {pm:.1f}°")
        return