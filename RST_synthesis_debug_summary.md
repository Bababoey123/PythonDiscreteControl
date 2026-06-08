# RST Synthesis — Debug & Changes Summary

This file summarizes recent changes made while debugging and improving the RST synthesis and simulation workflow.

## What I changed
- Added (in-notebook) a helper `make_desired_tf_compatible()` to construct desired closed-loop transfer functions that are numerically compatible with the RST algebraic solves.

## Root cause of the original issue
- The plant discrete numerator `B` was very small (order 1e-5) and the desired numerator `B_cl` was incompatible with `B`.
- Solving `B * T = B_cl` then required dividing by tiny numbers, producing extremely large `T` coefficients (and therefore large `R` gains).
- In addition, the plant had a near-singular denominator structure (double pole at `z=1`), which amplifies sensitivity and makes direct polynomial inversion ill-conditioned.

## How the fixes help
- Enforcing `nT = max(1, ...)` avoids negative-dimension arrays.
- Adding diagnostics around the least-squares solves (`la.lstsq`) makes it easier to see matrix shapes and identify under/over-determined cases.
- The `make_desired_tf_compatible()` helper constructs `A_cl`/`B_cl` so the Diophantine solve is well-determined and the DC gain is scaled to avoid huge `T` values.


2. (Optional) Replace the desired TF construction in your notebook with the provided `make_desired_tf_compatible()` helper and re-run `utils.Compute_Desired_RST`.

## Recommendations / next steps
- Use a Desired TF whose closed-loop DC gain is scaled relative to the plant DC gain (the helper does this).
- Avoid placing desired closed-loop poles too close to the origin (very fast) when the plant has poles at or near `z=1`.
- Consider adding Tikhonov (ridge) regularization to the least-squares solves to limit controller coefficient magnitudes.
- Optionally implement reference ramping/queuing in `Control/RSTController.py` (I prototyped this earlier).

## Files touched or added
- `Utils/utils.py` — fixed `nT` computation and added diagnostics.
- Notebook: added `make_desired_tf_compatible()` helper (in `PythonDiscreteLib.ipynb`).

If you want, I can create a short unit-test that asserts `|T| < T_max` for a set of admissible Desired TFs, or implement a regularized solver. Which would you prefer?
