# varpro-cpp

A **single-header, dependency-free C++17** implementation of variable projection
for separable nonlinear least squares. Copy
[`include/varpro/varpro.hpp`](include/varpro/varpro.hpp) into your project and
include it. There is no library to link and **no Eigen requirement**.

## Problem

Given observations $Y\in\mathbb{R}^{m\times s}$, a nonlinear parameter
vector $\alpha\in\mathbb{R}^q$, and a basis
$\Phi(\alpha)\in\mathbb{R}^{m\times n}$, solve

$$
\min_{\alpha,C}\;\|W(Y-\Phi(\alpha)C)\|_F^2,
\qquad W=\operatorname{diag}(w_1,\ldots,w_m).
$$

Variable projection eliminates the linear coefficients $C$ at every
nonlinear trial point. With $A=W\Phi(\alpha)$ and $B=WY$, it uses
$C=A^+B$ (a truncated minimum-norm SVD solve), then minimizes
$\|B-AC\|_F^2$ over $\alpha$ alone. All $s$ datasets share the
same nonlinear parameters and basis.

## Install and build

The only language requirement is **C++17** and its standard library.

```cpp
#include <varpro/varpro.hpp>
```

For a plain compiler invocation:

```sh
g++ -std=c++17 -O2 -Ipath/to/varpro-cpp/include your_program.cpp -o your_program
```

Or use CMake's header-only interface target:

```cmake
add_subdirectory(path/to/varpro-cpp)
target_link_libraries(your_target PRIVATE varpro::varpro)
```

To build the supplied example and test suite:

```sh
cmake -S . -B build -DBUILD_TESTING=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

An installed CMake package also provides `varpro::varpro` via
`find_package(varpro CONFIG REQUIRED)`.

## Example

```cpp
#include <varpro/varpro.hpp>
#include <cmath>

int main() {
  using varpro::Index;
  using varpro::Matrix;
  using varpro::Vector;

  const Vector t = Vector::LinSpaced(21, 0.0, 10.0);
  varpro::Problem p;
  p.observations.resize(t.size(), 1);
  for (Index i = 0; i < t.size(); ++i)
    p.observations(i, 0) = 3.0 * std::exp(-t(i) / 2.0) + 0.5;

  p.basis = [t](const Vector& a) {
    Matrix phi(t.size(), 2);
    for (Index i = 0; i < t.size(); ++i) {
      phi(i, 0) = std::exp(-t(i) / a(0));
      phi(i, 1) = 1.0;
    }
    return phi;
  };
  p.derivative = [t](const Vector& a, Index k) {
    Matrix d = Matrix::Zero(t.size(), 2);
    if (k == 0)
      for (Index i = 0; i < t.size(); ++i)
        d(i, 0) = std::exp(-t(i) / a(0)) * t(i) / (a(0)*a(0));
    return d;
  };

  auto fitted = varpro::fit(p, Vector{1.5});
  if (!fitted.converged()) return 1;
  // fitted.parameters(0) ~= 2.0; coefficients ~= [3.0, 0.5]
  return 0;
}
```

The accompanying
[`examples/double_exponential.cpp`](examples/double_exponential.cpp)
demonstrates two nonlinear parameters and multiple right-hand sides.

### Array interface

- `Vector(size)` and `Matrix(rows, cols)` allocate zero-initialized values.
- `vector(i)` and `matrix(i, j)` index elements; `Index` is `std::ptrdiff_t`.
- `Vector{1., 2.}` and `Matrix::Zero(m,n)` are supported.
- `Matrix` stores data **column-major**. `matrix.data()` uses that order,
  including the stacked residual Jacobian; `matrix << ...` fills **row-major**.
- Basic matrix/vector arithmetic, norms, `transpose()`, `col(j)` (a **copy**),
  and `leftCols(k)` are provided. This is intentionally **not** a complete
  Eigen expression API.

**Migration from Eigen:** `varpro::Matrix` and `varpro::Vector` are now
standalone value types. Code depending on Eigen expression operations (`.array()`,
row/column views, `.noalias()`, etc.) must use explicit loops or convert
at the application boundary. The `Problem`, `Evaluation`, `Options`, and
`FitResult` concepts and their shapes remain the same.

## API and numerical contracts

- `evaluate(problem, alpha, linear_options, jacobian_mode)` returns linear
  coefficients $n\times s$, weighted residuals $m\times s$, Jacobian
  $(ms)\times q$, retained rank, and singular-value diagnostics.
- `fit(problem, initial_alpha, options)` runs a damped Levenberg–Marquardt
  iteration in the $q$ nonlinear variables. It returns an evaluation
  consistent with the final parameters, including on evaluation-budget exit.
- `m > 0`, `s > 0`, `0 < n <= m`, `q > 0`; for `fit` also `ms >= q`.
  Observations, parameters, basis, derivatives, and weights must be finite.
- Weights are optional, nonnegative **residual multipliers** shared across
  datasets. Use `1/sigma`, **not** `1/(sigma*sigma)`, for known standard
  deviations. At least one weight must be positive if weights are specified.
- The basis width must remain fixed during a fit; each derivative has the
  exact shape of the basis.
- `LinearOptions::rcond` selects which singular directions to keep:
  $\sigma_i/\sigma_{\max}>\mathrm{rcond}$. The default negative value
  selects $\max(m,n)\epsilon$. Rank zero is valid.
- `JacobianMode::kaufman` (default) computes
  $- (I-U_rU_r^T)D_kC$. `retained_subspace` adds
  $-U_r\Sigma_r^{-1}V_r^TD_k^TR$. The alias `exact` means
  `retained_subspace`, **not** the derivative of a repeatedly truncated SVD
  when discarded nonzero singular directions rotate.

### Internal numerical methods

The linear solve uses a **scaled, one-sided Jacobi SVD**, directly
orthogonalizing columns. It does not form $A^TA$, so rank decisions avoid
squaring the condition number. The LM step solves an **augmented damped
least-squares system** by the same SVD, with a relative step trust region.
This is a new optimizer implementation; its trajectory and termination counts
may differ from the old Eigen LM implementation.

This design favors small or moderate basis widths and nonlinear parameter
counts. It does not promise global minimization, constraints on nonlinear
parameters, or the speed of a vendor-optimized LAPACK implementation on very
large dense systems. For a difficult fit, examine the reported rank, retained
condition estimate, residual, and `Status` rather than interpreting
`converged()` as proof that the target parameters were identified.

For implementation invariants and edge cases, see [CPP_DESIGN.md](CPP_DESIGN.md).

## Attribution

This port derives from [Geo's Rust `varpro`](https://github.com/geo-ant/varpro),
which credits Dianne P. O'Leary and Bert W. Rust for
[Variable projection for nonlinear least squares problems](https://doi.org/10.1007/s10589-012-9492-9)
(2013). It also draws on the [VarPro articles in Geo's Notepad](https://geo-ant.github.io/blog/2020/variable-projection-part-1-fundamentals/)
and the underlying work of Golub and Pereyra on separable nonlinear least
squares. The original acknowledgement and historical background are preserved
in [`docs/original/README.md`](docs/original/README.md).

MIT licensed; see [LICENSE](LICENSE).
