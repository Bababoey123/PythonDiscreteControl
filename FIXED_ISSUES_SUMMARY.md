# Fixed Issues Summary

This file documents the issues identified and resolved during the recent investigation into RST synthesis and discrete/hybrid simulation timing.

## 1. `Utils/utils.py` RST solver type issue
- Identified a problematic unpacking of `scipy.linalg.lstsq` results in `Compute_Denominator_Matching_RST`.
- The code used `theta, *_ = la.lstsq(M, A_cl)`, which caused a type-check warning because the return type is not iterable in the static analysis.
- Fixed by preserving the existing solver call and adding `# type: ignore[assignment]` to the unpacking line.
- Result: `Utils/utils.py` now passes error checking for that statement.

## 2. RST synthesis semantics clarification
- Confirmed the discrete closed-loop RST structure is solved using the true plant numerator `B` and normalized plant denominator `A`.
- Ensured the Diophantine equation `A S + B R = A_cl` is assembled without incorrectly scaling `B` by DC gain.
- This preserves correct controller synthesis semantics for denominator-matching closed-loop design.

## 3. Discrete vs hybrid control timing
- Compared `run_discrete_control` and `run_continuous_control_loop` control-update order.
- Verified both loops use the current measured output to compute `u[k]` before applying it to the plant.
- The apparent difference comes from logging and intermediate plant integration, not from a timing bug in the control law.

## Files changed
- `Utils/utils.py`

## Notes
- The investigation focused on code behavior and control timing semantics.
- No additional file changes were made for the hybrid control path during this summary generation.
