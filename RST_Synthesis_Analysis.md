# RST Synthesis Analysis

## Notebook RST synthesis sections

The notebook uses two main RST synthesis flows:

1. `Compute_Denominator_Matching_RST(A_cl, model.Tf_dis)`
   - Used in the pole-placement example with `A_cl` built from `utils.poles_to_denominator(poles)`.
   - Produces `S`, `R`, and `T` for a denominator-matching design.
   - Then constructs an `RSTController(R, S, T)` and simulates with `run_discrete_control`.

2. `Compute_Desired_RST(Desired_TF, plant.Tf_dis)`
   - Used after generating a desired transfer function with `utils.Place_real_radius(...)`.
   - Produces `S_tf`, `R_tf`, `T_tf` that match a full desired closed-loop TF numerator and denominator.
   - Simulates with `run_continuous_control_loop` using `RSTController(R_tf, S_tf, T_tf)`.

## Control theory interpretation

### Plant model

The plant is represented as a discrete-time transfer function:

- `G(z) = B(z) / A(z)`

where:
- `A(z)` is the plant denominator polynomial in `z^{-1}` or descending `z` powers.
- `B(z)` is the plant numerator polynomial.

### RST control law

The implemented controller follows the standard RST form:

- `v[k] = T(z) r[k] - R(z) y[k]`
- `u[k] = (1/S(z)) v[k]`

This means the closed-loop transfer function from reference to output is:

- `H_cl(z) = B(z) T(z) / (A(z) S(z) + B(z) R(z))`

The denominator is the key closed-loop polynomial, and the numerator is shaped by `T(z)`.

### Diophantine equation

Both synthesis functions solve the Diophantine equation:

- `A(z) S(z) + B(z) R(z) = A_cl(z)`

This guarantees the closed-loop denominator equals the desired denominator polynomial `A_cl(z)`.

The code uses Toeplitz convolution matrices to form a linear system:

- `A S` and `B R` are written as convolution products.
- The unknown vector `[S; R]` is solved by least squares.
- Controller orders are chosen as:
  - `deg(S) = deg(B)`
  - `deg(R) = deg(A)`

This choice is a canonical RST structure for single-input single-output systems.

## Differences between the two synthesis functions

### `Compute_Denominator_Matching_RST`

- Primary objective: enforce the desired closed-loop denominator only.
- `T` is chosen after `S` and `R`.
- If `A_cl(1) != 0`, the code computes:
  - `T = A_cl(1) / B(1)`
  - This is effectively a reference scaling to try to preserve steady-state gain.
- If `A_cl(1) == 0` (a pole at `z = 1` or pure integrator):
  - The code cannot enforce a finite DC gain.
  - It now sets `T = 1` and warns that the DC gain is undefined or infinite.
- The consequence is:
  - denominator shaping is preserved,
  - but the closed-loop reference gain is not a finite unity gain in the integrator case.

### `Compute_Desired_RST`

- Primary objective: match a full desired transfer function `Desired_TF`.
- It extracts both `A_cl` and `B_cl` from the desired TF.
- It solves the same Diophantine equation for `S`, `R`.
- Then it computes `T(z)` by exact polynomial division:
  - `B(z) T(z) = B_cl(z)`
- This is the correct full-numerator matching step when the numerator can be obtained by multiplying `B` by `T`.
- If `B_cl` is not divisible by `B`, the function raises a `ValueError`.

## Consistency issues in the notebook

1. Mixed design goals
   - The notebook uses `Compute_Denominator_Matching_RST` for denominator-only design and `Compute_Desired_RST` for full TF matching.
   - These are different design objectives. Their results are not directly comparable unless the intended closed-loop behavior is clearly defined.

2. DC gain expectations
   - `Compute_Denominator_Matching_RST` is not guaranteed to produce a finite or unity DC gain when `A_cl` has a unit-circle pole.
   - The notebook should not interpret a pole-at-1 design as having a well-defined steady-state gain.

3. `T` interpretation
   - In denominator-matching mode, `T` is effectively a constant reference scaling unless `A_cl(1) == 0`.
   - In desired-TF mode, `T(z)` can be a higher-order polynomial exactly matching the desired numerator.

4. Potential naming confusion
   - The notebook reuses `rst_3` and `rst_2` for different experiments.
   - This is not a functional bug, but it can make the design flow harder to follow.

## Recommended interpretation for the notebook

- If you want pole placement with unspecified numerator behavior, use `Compute_Denominator_Matching_RST`.
- If you want a specific desired closed-loop TF (shape and gain), use `Compute_Desired_RST`.
- If `A_cl` includes a pole at `z=1`, treat the result as marginally stable, not as a gain-matched design.
- For comparison between simulation and theory, compare the closed-loop denominator directly and check whether `B(z) T(z)` matches the desired numerator.

## Practical note

The notebook’s RST controller implementation and the synthesis utilities are consistent with standard discrete-time RST theory, but the two functions serve different roles:

- `Compute_Denominator_Matching_RST` = denominator-only shaping
- `Compute_Desired_RST` = full desired transfer function matching

Use the appropriate function for the design goal, and do not assume both give the same closed-loop gain behavior.
