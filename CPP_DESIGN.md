# C++17 header-only design and verification contract

## Scope

One public header (`include/varpro/varpro.hpp`), no implementation object,
no external math packages, no global mutable state. All non-template
definitions are `inline`, so the header can be included in multiple
translation units without one-definition-rule conflicts.

The public matrix and vector are compact value types. Matrix storage is
column-major; comma initialization is row-major. The `Problem` callbacks
take `Vector` and return `Matrix` by value. Their types and code do not
require Eigen.

## Algebraic separation

For $A = W\Phi(\alpha)$, $B=WY$, and retained thin SVD
$A \approx U_r\Sigma_r V_r^T$:

1. $C=V_r\Sigma_r^{-1}U_r^T B$.
2. $R=B-AC$.
3. $J_k=-(I-U_rU_r^T)D_k C$ for Kaufman.
4. Add $-U_r\Sigma_r^{-1}V_r^TD_k^TR$ for the retained-subspace formula.

The same rank and factors are reused for all four operations, including
rank-deficient bases and explicit truncation. Rank zero yields zero
coefficients, $R=B$, and a zero Jacobian.

## Numerics

- **SVD:** one-sided Jacobi on the globally scaled matrix $A/a_{max}$.
  Rotation orthogonalizes columns and accumulates right singular vectors.
  The resulting singular values are sorted descending, then rescaled. The
  zero matrix is handled without division. Failure to converge is reported.
- **Rank:** retain precisely $\sigma_i/\sigma_{max}>\mathrm{rcond}$, with
  automatic `rcond=max(m,n)*epsilon`. The condition estimate covers only the
  retained subspace.
- **LM:** at each trial, eliminate $C$. For residual Jacobian $J$,
  gradient $g=J^Tr$, and diagonal $D$ based on squared Jacobian column
  norms, solve the augmented least-squares system

  $$
  \begin{bmatrix} J\\\sqrt{\lambda D}\end{bmatrix}\delta
  \approx \begin{bmatrix}-r\\0\end{bmatrix}.
  $$

  Trial acceptance uses actual-to-predicted cost reduction. Damping increases
  after rejection and decreases after a successful step. The scaled step is
  limited to $\|\delta_i/(1+|\alpha_i|)\|_2\le 1/2$ to prevent jumps far
  outside the local linear model.
- **Termination:** `gtol` checks a normalized Jacobian/residual cosine,
  `xtol` checks the parameter step, `ftol` checks accepted relative cost
  reduction. Evaluation budget counts the initial and each trial residual;
  the final Jacobian evaluation is not counted. Iterations start at one.
- **Failure behavior:** invalid user input throws `invalid_argument`;
  nonfinite callback/numerical results throw `domain_error`; exceptions
  from callbacks propagate. `Status::stalled` marks unsuccessful damping
  after repeated rejection.

## Required tests

1. Exact single and multiple right-hand-side fits with matching linear
   coefficients; both Jacobian modes.
2. Weighted problem against explicit preweighting; weighted orthogonality.
3. Residual Jacobians against centered finite differences; objective
   derivative against $2\langle r,J_k\rangle$.
4. Duplicate columns and zero basis: minimum-norm coefficients and
   retained projector; explicit/automatic singular-value cutoff.
5. Very large (finite) basis values and strongly scaled directions.
6. Rotating truncated subspace: document why retained-subspace Jacobian
   differs from the derivative of the re-truncated residual map.
7. Reproducible randomized small-matrix derivative and projector tests.
8. Bad shapes, weights, nonfinite inputs/results, and strict evaluation budget.
9. One-header use, multiple translation units, installed CMake target,
   GCC/Clang/MSVC Debug/Release, and Linux address/undefined sanitizers.

The automated suite resides in `tests/cpp/test_varpro.cpp` and
`.github/workflows/ci.yml`. The finite-difference checks validate local
derivatives where the retained rank is constant. They do not establish
global identifiability or convergence from arbitrary initial guesses.
