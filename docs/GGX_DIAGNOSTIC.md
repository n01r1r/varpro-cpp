# Conditional GGX noisy-stage diagnostic

## Scope and decision

Reviewed `main` at `c94f0a32b27b23ec898a2fe7f133d369467fbbee` against
the README and `CPP_DESIGN.md`. Reuse the compact-SVD coefficient solve and
shared-RHS projection, but keep the affine offset and bounded branch outside
`Problem`/`fit`. No public API or solver changes.

The example assumes calibrated linear neutral intensities, known unit lights,
fixed normal/view and fixed weights. The normal and chromatic diffuse deviation
have already been supplied by stage one. It uses Lambert diffuse, anisotropic
GGX, Schlick F0=0.04 and G=1; it does not test the validity of that approximation
on real materials. Formula reference: [PBRT 4, section 9.6](https://www.pbr-book.org/4ed/Reflection_Models/Roughness_Using_Microfacet_Theory).
PBRT includes masking; this example deliberately uses the reduced G=1 model.

For fixed n, write y_k = a_k mu + s_k(theta) + noise,
a_k = (n dot l_k)/pi, and theta = (log alpha_t, log alpha_b, phi).
In a fixed tangent frame rotated by phi, with halfway coordinates (x,y,z),

```math
Q_k=x_k^2/\alpha_t^2+y_k^2/\alpha_b^2+z_k^2,\qquad
s_k=\frac{F_k}{4\pi(n\cdot v)\alpha_t\alpha_b Q_k^2}.
```

Physical coordinates, 0 < alpha_t, alpha_b <= 1, build a compatible GGX
matrix by construction: lambda=sqrt(4*pi*(n dot v)*alpha_t*alpha_b),
S=lambda*R*diag(alpha_t^-2,alpha_b^-2)*R^T. Thus S >= lambda I and
lambda^2 det(S) = [4*pi*(n dot v)]^2. No unconstrained four-coefficient
fit is substituted for this physical parameterization. Log coordinates
include the chain rule; log alpha <= 0 is checked, not enforced by an optimizer.

## Elimination and derivative

With A=Wa and b(theta)=W[y-s(theta)],

```math
\mu_{free}=\frac{A^\top b}{A^\top A},\quad
r=b-A\mu_{free},\quad J_{red}=-(I-AA^+)W\,\partial_\theta s.
```

The adapter evaluates the constant one-column basis a against four RHS:
`[y-s, -ds/dtheta_0, -ds/dtheta_1, -ds/dtheta_2]`. One SVD supplies the
mean and all three projections. This is algebraic reuse of multiple RHS;
the derivative columns are not independent observed datasets. The returned
library `Evaluation::jacobian` is zero because the basis is constant and
the RHS are fixed within each call. The changing-offset derivative is the
last three **residual columns**, assembled by the adapter. Passing a frozen
shifted observation matrix to `fit` would solve the wrong objective.

For supplied chromatic deviation Delta rho, bounds are
lower=max_c(-Delta rho_c), upper=min_c(1-Delta rho_c).
The caller supplies these bounds; the example uses [0.2,0.8]. For fixed bounds,

```math
\mu_* = \operatorname{clip}(\mu_{free},\ell,u),\qquad
r=W[y-s-a\mu_*].
```

Strictly inside the interval, use the projected derivative above. Strictly
outside it, the active mean is constant and J_red=-W ds/dtheta. At an
active-set switch there need not be a common two-sided residual derivative;
`at_switch` flags proximity within a roundoff tolerance. The returned matrix
there is only a branch derivative. If lower=upper, the mean is always fixed.
Data-derived bounds or an estimated normal being updated require additional
derivatives and are outside this example. Do not enforce y-a*mu>0 on every
noisy sample: the predicted physical s is positive, but noisy samples may not be.

If n is optimized jointly, both A and the offset change. On an interior,
constant-rank branch the required derivative is

```math
\partial_k r=P_A^\perp(b_k'-A_k'\mu)
              -(A^+)^\top(A_k')^\top r.
```

The retained-subspace formula provides the two A-dependent terms, but the
current API has no b'_k callback. This example does not claim to implement
that joint problem or propagate uncertainty from stage one.

## What each diagnostic means

| Quantity | Reuse / interpretation |
| --- | --- |
| `Evaluation::rank`, `diagnostics` | Rank and singular values of Wa only. A nonzero one-column basis has condition 1, however small its magnitude. Inspect sigma_max for absolute mean sensitivity. |
| `spectrum(J_red)` | Separate thin SVD using the README's strict relative cutoff. Reports all singular values, local numerical rank and retained condition in the stated log/log/radian coordinates. |
| `retained_subspace` | Existing constant-rank retained-direction contract remains unchanged. It does not account for a changing RHS. |
| Multiple observed RHS | Valid only when n, specular theta, lighting and weights are shared and each column may have its own mean. Spatially different SVBRDFs or RGB coupling are not automatically this model. |
| FD tests | Recompute the entire conditional mean/residual at each perturbation; check nonzero residuals and remain on one active branch for derivative agreement. |

Interior rank(J_red) <= min(3,K-1); at an active mean bound it can be as high
as min(3,K). Full column rank is a sufficient local first-order regularity
test, not global uniqueness. Rank deficiency indicates a missing first-order
direction; it alone does not prove nonlinear nonidentifiability. At isotropy,
however, s is independent of phi for every angle: that direction is truly
unobservable in these coordinates. Axis sign, angle periodicity and axis
permutation also remain global equivalences.

For appropriately whitened Gaussian noise and fixed stage-one quantities,
J_red^T J_red is the conditional local information matrix (Schur complement
of the free mean in the interior). Raw condition numbers change under parameter
rescaling. Do not turn them into uncertainty estimates without the noise model
and the effect of stage-one estimation. A finite retained condition can coexist
with rank < 3; rank-zero condition 0 is a sentinel, not good conditioning.

## Counterexamples and exclusions

1. **Free specular gain:** replacing the affine model by Phi=[a,s(theta)]
   solves both mu and a gain beta. For independent a and s, data y=a*mu+2*s
   fit exactly with beta=2; the fixed-gain physical model at the same theta
   has nonzero residual. The test suite checks this counterexample.
2. **Free GGX coefficients:** fitting H*(A,B,C,lambda) does not impose GGX
   compatibility. At n dot v=1, lambda=1 and S=I satisfy positive eigenvalues
   and alpha=1 but give lambda^2 det(S)=1 instead of (4*pi)^2.
3. **Transformed observations:** q(mu)=sqrt(F/(y-a*mu)) is nonlinear in mu;
   mu is not a linear coefficient of H. This transform also amplifies noise
   with derivative -sqrt(F)/(2*s^(3/2)). Intensity LS and transformed LS
   have different noise/objective contracts.
4. **Truncated rotating subspace:** as already tested by `varpro_tests`,
   discarding a nonzero singular direction can make retained-subspace J differ
   from re-truncated residual FD even at constant numerical rank. The example's
   one-column positive basis has no such discarded nonzero direction.
5. **Rank transition:** the generic diag(1,t) test crosses the strict cutoff
   at t=0.01, rcond=0.01. Its residual jumps; central FD is not a smooth
   Jacobian check. This differs from smooth GGX isotropy, where J loses rank
   without a residual discontinuity. Diagnostic truncation of J does not
   truncate the physical residual or its derivative.

## Verification

Build and run with the README CMake commands. `ggx_diagnostic_tests` is
independent of `VARPRO_BUILD_EXAMPLE`, so example-disabled/sanitizer builds
also exercise the adapter. Checks remain active in Release. Tests cover
weighted mean elimination, nonzero-residual FD at three step sizes, log chain
rules, objective gradients, MRHS/manual preweight equivalence, both active
bounds, switch kink, anisotropic/isotropic and cutoff-dependent local rank,
rank-zero sentinel, incorrect free-gain mapping, and linear cutoff crossing.
These are implementation and diagnostic checks, not recovery experiments.

Verified on 2026-10-01 with Linux GCC 13.3.0, Eigen 3.4.0 and CMake 4.4.3:

- Release build: CTest 2/2 passed; both documented examples exited zero.
- Debug ASan/UBSan, `VARPRO_BUILD_EXAMPLE=OFF`: CTest 2/2 passed with
  `ASAN_OPTIONS=detect_leaks=0`. LeakSanitizer could not enumerate `/proc`
  threads in this execution environment; leak detection was not validated.
- `git diff --check`: clean. Other compilers/platforms were not run locally.

The noisy anisotropic fixture printed mu=0.399792660266, linear rank=1 and
linear retained condition=1. In log/log/radian coordinates, J_red singular
values were (0.0333640347174, 0.0248882455168, 0.0231529340633), rank=3 and
retained condition=1.44102836496 at rcond=1e-10. Its weighted squared residual
was 9.12473930788e-05. At isotropy, the spectrum was
(0.0313643494674, 0.0283537888815, 0), rank=2. These numbers describe this
conditional fixture, not an identifiability theorem for arbitrary lighting.
