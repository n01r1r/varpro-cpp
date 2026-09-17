# varpro-cpp

Small C++17 implementation of variable projection for separable nonlinear
least-squares problems. The library has one public header, one implementation
file, and Eigen as its only C++ dependency.

## Acknowledgements and provenance

This project is a C++ implementation of the variable-projection workflow
from [Geo's `varpro` repository](https://github.com/geo-ant/varpro), the
original Rust library for fitting separable nonlinear models. The C++ port was
developed from the [C++ reference repository](https://github.com/n01r1r/varpro-cpp)
and is maintained here as a smaller Eigen-based implementation.

The original `varpro` README credits Professor
[Dianne P. O'Leary](http://www.cs.umd.edu/~oleary/) and Professor
[Bert W. Rust](https://math.nist.gov/~BRust/) for the paper that enabled the
author to understand variable projection and develop that implementation. We
preserve that acknowledgement here, including the cited paper:

- O'Leary, D. P., and Rust, B. W. [Variable projection for nonlinear least
  squares problems](https://doi.org/10.1007/s10589-012-9492-9), *Computational
Optimization and Applications* 54, 579–593 (2013).

The original `varpro` project also records its gratitude to Professor O'Leary
for answering questions about the paper and implementation details.

We also acknowledge Geo's explanatory articles in [Geo's
Notepad](https://geo-ant.github.io/), which provide the implementation-oriented
derivation used by this port:

- [The Variable Projection Method – Nonlinear Least Squares Fitting with
  VarPro](https://geo-ant.github.io/blog/2020/variable-projection-part-1-fundamentals/)
  (updated 2023);
- [Global Fitting of Multiple Right Hand Sides with Variable
  Projection](https://geo-ant.github.io/blog/2024/variable-projection-part-2-multiple-right-hand-sides/).

The first article notes errors in some formulas in the O'Leary–Rust
presentation and presents corrected formulas. For the broader mathematical
foundation, see also Golub and Pereyra, [Separable nonlinear least squares: the
variable projection method and its applications](https://doi.org/10.1088/0266-5611/19/2/201).

## What it does

Many models contain two different kinds of parameters:

- **Nonlinear parameters** change the shape of the basis functions. Examples
  include decay times, frequencies, or peak locations.
- **Linear coefficients** only scale and combine those basis functions.

For example, a sum of exponentials can be written as

```math
y(t) = c_1 e^{-t / \tau_1} + c_2 e^{-t / \tau_2} + c_0.
```

The time constants `tau1` and `tau2` are nonlinear parameters. The
coefficients `c1`, `c2`, and `c0` are linear parameters.

Variable projection takes advantage of this separation. In matrix form, the
model is

```math
Y \approx \Phi(\alpha) C,
```

where:

| Symbol | Shape | Meaning |
| --- | --- | --- |
| `Y` | `m x s` | Observations; each column is one dataset. |
| `Phi(alpha)` | `m x n` | Basis matrix evaluated at nonlinear parameters `alpha`. |
| `C` | `n x s` | Linear coefficients; each dataset has its own coefficient column. |
| `alpha` | `q` | Nonlinear parameters shared by all datasets. |

With optional shared weights `W = diag(w_1, ..., w_m)`, the fitted problem is

```math
\min_{\alpha, C} \left\| W\left(Y - \Phi(\alpha)C\right) \right\|_F^2.
```

For a fixed `alpha`, the best `C` is a linear least-squares solve. The
library computes it with a numerical-rank compact SVD, then gives only
`alpha` to the nonlinear optimizer:

```math
A(\alpha) = W\Phi(\alpha), \qquad B = WY,
```

```math
C(\alpha) = \mathop{\arg\min}_C \left\|B - A(\alpha)C\right\|_F^2
           = A(\alpha)^+B,
```

The retained decomposition is

```math
A(\alpha) \approx U_r\Sigma_rV_r^\top,
\qquad r = \text{numerical rank},
```

and the same retained rank is used for the minimum-norm solve and the range
projector:

```math
A^+ = V_r\Sigma_r^{-1}U_r^\top,
\qquad P_A^\perp = I - U_rU_r^\top.
```

```math
\min_{\alpha} F(\alpha), \qquad
F(\alpha) = \left\|B - A(\alpha)C(\alpha)\right\|_F^2.
```

In practical terms, each iteration is:

1. evaluate the basis and its derivatives at a trial `alpha`;
2. solve the linear coefficients with SVD;
3. form the weighted residual and the selected residual Jacobian;
4. let Levenberg–Marquardt update only `alpha`.

This is useful when the model is a linear combination of nonlinear basis
functions and analytical basis derivatives are available. It is not a general
optimizer for models in which every parameter is nonlinear.

## Jacobian modes

The default `JacobianMode::kaufman` uses the compact-SVD Kaufman approximation.
For a residual `R = B - A*C`, `JacobianMode::exact` adds the linear-coefficient
response without constructing a pseudoinverse:

```math
J_k^{\text{exact}} = -P_A^\perp D_kC
 - U_r\Sigma_r^{-1}V_r^\top D_k^\top R.
```

The minus sign in the second term follows the library's `B - A*C` residual
convention. `JacobianMode::exact` is the full VarPro Jacobian for a locally
constant retained rank, with discarded singular directions treated as null.
The exact mode can be selected for an evaluation or a fit:

```cpp
const auto evaluation = varpro::evaluate(problem, alpha,
                                         varpro::JacobianMode::exact);

varpro::Options options;
options.jacobian_mode = varpro::JacobianMode::exact;
options.linear_options.rcond = 1e-10;  // < 0 keeps the automatic cutoff.
const auto result = varpro::fit(problem, initial, options);
```

## Build and run

Requirements:

- CMake 3.16 or newer;
- a C++17 compiler;
- Eigen 3.4 or newer, including its bundled `unsupported` headers.

No Rust compiler, BLAS, LAPACK, Python, or additional nonlinear-solver
library is required. With Eigen installed and discoverable by CMake:

```sh
cmake -S . -B build
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

For a single-configuration generator, add
`-DCMAKE_BUILD_TYPE=Release` to the configure command.

If Eigen is only available as a source checkout, point CMake at the directory
that contains both `Eigen/` and `unsupported/`:

```sh
git clone --depth 1 --branch 3.4.0 https://gitlab.com/libeigen/eigen.git build/_deps/eigen-src
cmake -S . -B build -DEIGEN3_INCLUDE_DIR=build/_deps/eigen-src
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

The example fits two double-exponential curves. Run
`build/Release/double_exponential.exe` with Visual Studio, or
`build/double_exponential` with a single-configuration generator. It prints
the shared time constants, the coefficient column for each curve, and the
squared weighted residual error.

## Minimal example

The following example fits one curve to

```math
y(t) = 3e^{-t/2} + 0.5.
```

The basis has two columns: the exponential and a constant offset. Its one
nonlinear parameter is the decay time `alpha[0]`; both basis columns have
linear coefficients.

```cpp
#include <varpro/varpro.hpp>
#include <cmath>
#include <stdexcept>

varpro::Vector t = varpro::Vector::LinSpaced(50, 0.0, 10.0);
varpro::Problem problem;
problem.observations.resize(t.size(), 1);
for (Eigen::Index i = 0; i < t.size(); ++i) {
    problem.observations(i, 0) =
        3.0 * std::exp(-t[i] / 2.0) + 0.5;
}

// Return Phi(alpha), an m x n basis matrix.
problem.basis = [t](const varpro::Vector& alpha) {
    varpro::Matrix phi(t.size(), 2);
    phi.col(0) = (-t.array() / alpha[0]).exp().matrix();
    phi.col(1).setOnes();
    return phi;
};

// Return d Phi / d alpha[k], also an m x n matrix.
problem.derivative = [t](const varpro::Vector& alpha, Eigen::Index k) {
    varpro::Matrix dphi = varpro::Matrix::Zero(t.size(), 2);
    if (k == 0) {
        dphi.col(0) =
            ((-t.array() / alpha[0]).exp()
             * t.array() / (alpha[0] * alpha[0])).matrix();
    }
    return dphi;
};

varpro::Vector initial(1);
initial << 1.5;
const varpro::FitResult result = varpro::fit(problem, initial);

if (!result.converged()) {
    // Inspect result.status and decide whether the fit is usable.
    throw std::runtime_error("fit did not converge");
}

// Approximately [2], [3, 0.5]^T, and a near-zero residual for this exact data.
const varpro::Vector& alpha = result.parameters;
const varpro::Matrix& coefficients = result.evaluation.coefficients;
const varpro::Matrix& residuals = result.evaluation.residuals;
```

The callback contract is intentionally explicit:

- `basis(alpha)` returns the same `m x n` shape for every evaluation;
- `derivative(alpha, k)` returns the partial derivative
  `d Phi(alpha) / d alpha[k]` with that same shape. If a transformed parameter
  is used, this derivative must include the corresponding chain rule;
- the parameter order in `initial` is the order used by both callbacks.

The library does not impose parameter bounds. For a quantity that must be
positive, use a transformed parameter in the callback, such as
`tau = std::exp(log_tau)`, and optimize `log_tau`.

## Multiple datasets (multiple right-hand sides)

Store datasets as columns of one matrix:

```cpp
varpro::Matrix observations(m, number_of_datasets);
// observations.col(j) is dataset j.
problem.observations = observations;

const varpro::FitResult result = varpro::fit(problem, initial);
// result.parameters: q
// result.evaluation.coefficients: n x number_of_datasets
// result.evaluation.residuals: m x number_of_datasets
```

All columns use the same `Phi(alpha)` and therefore share the nonlinear
parameters. Each column still receives its own linear coefficient vector. The
fit minimizes the total error over all columns:

```math
F(\alpha) = \sum_{j=1}^{s}
\left\|W\left(y_j - \Phi(\alpha)c_j\right)\right\|_2^2.
```

This lets related curves be fitted globally while allowing their amplitudes or
offsets to differ.

## Weights

`problem.weights` is an optional vector of `m` shared residual multipliers.
With standard deviations `sigma`, use inverse standard deviations:

```cpp
problem.weights = sigma.cwiseInverse();  // 1 / sigma, not 1 / sigma^2
```

An empty vector means unit weights. Weights must be finite and nonnegative,
with at least one positive entry. They multiply residuals before the squared
error is calculated, so the reported `residuals` matrix is
`W * (Y - Phi * C)`.

## Inspecting an evaluation

Use `evaluate(problem, alpha)` when the nonlinear parameters are supplied by
another optimizer or when you want to inspect one point without running a
fit:

```cpp
const varpro::Evaluation evaluation = varpro::evaluate(problem, alpha);

evaluation.coefficients;  // SVD linear solve, n x s
evaluation.residuals;     // weighted residuals, m x s
evaluation.jacobian;       // selected residual Jacobian, (m * s) x q
evaluation.rank;           // retained SVD rank
evaluation.squared_error();// sum of squared residual entries
```

## Results, errors, and numerical details

`fit` returns the final accepted parameter vector and the corresponding
coefficients and residuals in one `FitResult`. Check `result.converged()` (or
inspect `result.status`) before treating a fit as successful. The status is one
of:

- `converged`: Levenberg–Marquardt met a local stopping criterion;
- `evaluation_limit`: the residual evaluation budget was exhausted;
- `stalled`: a stopping tolerance became too small to make progress;
- `numerical_failure`: the optimizer or a numerical calculation failed.

Convergence is local: it does not establish that the solution is unique or
globally optimal. `Options` controls the maximum number of evaluations and the
LM tolerances `ftol`, `xtol`, and `gtol`.

The implementation uses real, double-precision matrices. At each parameter
evaluation it computes a numerical-rank compact SVD of `W * Phi(alpha)`,
truncates singular values according to `sigma_i > rcond * sigma_max`, and uses
the retained subspace for the minimum-norm linear solution and projector.
`LinearOptions::rcond < 0` selects the automatic default
`max(m, n) * epsilon`; a nonnegative value makes the numerical rank an explicit
modeling choice.

The returned Jacobian uses `Options::jacobian_mode` for `fit`, and Kaufman is
the default for `evaluate`. Kaufman is generally not the exact residual
derivative when the residual is nonzero. With full numerical rank, the exact
mode is tested directly against residual finite differences at nonzero
residuals. If an explicit `rcond` truncates a nonzero singular value, exact is
not the derivative of the residual map that re-truncates the original basis at
each parameter value; the discarded direction is treated as null by this
contract. A locally constant numerical rank alone does not remove this
distinction. Derivatives may be discontinuous where the numerical rank changes.
See
[CPP_DESIGN.md](CPP_DESIGN.md) for the precise formulas, cutoff, validation
rules, and verification evidence.

Invalid shapes, missing callbacks, nonfinite inputs, negative or all-zero
weights, and invalid options throw `std::invalid_argument`. Nonfinite model
values or numerical outputs throw `std::domain_error`. Exceptions raised by a
callback propagate to the caller; an invalid trial evaluation aborts the fit.

Linear coefficients are unconstrained. Nonnegative coefficients, NNLS, and
linear coefficient bounds are not supported by this compact VarPro API. A
Tikhonov penalty can be represented without a new solver by augmenting the
linear system. For a fixed `L` with `p` rows, use `[A; lambda * L]` and
`[B; 0]` in the callbacks (and append zero derivative rows):

```cpp
const Eigen::Index m = problem.observations.rows();
const Eigen::Index p = L.rows();
const Eigen::Index n = L.cols();
varpro::Problem augmented = problem;
augmented.observations.conservativeResize(m + p, problem.observations.cols());
augmented.observations.bottomRows(p).setZero();
augmented.basis = [base = problem.basis, L, lambda, m, p](const varpro::Vector& x) {
    varpro::Matrix result(m + p, L.cols());
    result.topRows(m) = base(x);
    result.bottomRows(p) = lambda * L;
    return result;
};
augmented.derivative = [base = problem.derivative, m, p, n](const varpro::Vector& x,
                                                            Eigen::Index k) {
    varpro::Matrix result = varpro::Matrix::Zero(m + p, n);
    result.topRows(m) = base(x, k);
    return result;
};
```

The augmentation example assumes an unweighted or already preweighted
`problem`; shared observation weights should be incorporated consistently into
the top block before appending the penalty rows.

## CMake integration

Install the library and use the exported CMake target:

```sh
cmake --install build --config Release --prefix /path/to/prefix
cmake -S consumer -B consumer/build -DCMAKE_PREFIX_PATH=/path/to/prefix
```

The installed package provides `varpro::varpro`. The consumer must provide an
Eigen 3.4+ CMake package. For a source-tree consumer, use `add_subdirectory`
and link the same target directly. Consumers can disable the example and tests
with `-DVARPRO_BUILD_EXAMPLE=OFF` and `-DBUILD_TESTING=OFF`.

## Why Eigen stays

Eigen supplies both the SVD and the bundled
`unsupported/Eigen/LevenbergMarquardt` implementation. Keeping those numerical
building blocks in a maintained dependency makes this port smaller and easier
to review. No speedup over Rust or equivalence of optimizer trajectories is
claimed.

## Provenance and references

- [Original `varpro` repository](https://github.com/geo-ant/varpro).
- [C++ reference repository](https://github.com/n01r1r/varpro-cpp) at reference commit
  `29d847047966c0dd93550d7c5222bdfa34befdea`.
- [Archived original README](docs/original/README.md), retained verbatim for
  provenance and the original citations.
- [Detailed port references and provenance](docs/REFERENCES.md).

The maintained tree contains the C++ implementation only. The archived README
preserves the upstream project's original citations and acknowledgement, and
the original MIT license and attribution are retained in [LICENSE](LICENSE).
