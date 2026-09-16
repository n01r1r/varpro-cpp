# varpro-cpp

Small C++17 variable projection for separable nonlinear least squares.
One header, one implementation file, and Eigen as the only dependency.

Given a basis matrix `Phi(alpha)` and observations `Y`, the library solves

```text
minimize || W * (Y - Phi(alpha) * C) ||_F^2
```

It solves the linear coefficients `C` by SVD at each `alpha`, then optimizes
only `alpha` with Levenberg-Marquardt. One observation column fits one curve;
several columns perform a global fit with shared nonlinear parameters and
separate linear coefficients. All columns share the same basis and weights.

## Build and run

Requires CMake 3.16+, a C++17 compiler, and Eigen 3.4 including its bundled
`unsupported` headers. Tested dependency version: Eigen 3.4.0. No Rust compiler,
BLAS, LAPACK, Python, or additional nonlinear solver library is required.

With Eigen installed and discoverable by CMake:

```sh
cmake -S . -B build
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

Install the library and consume it from another CMake project:

```sh
cmake --install build --config Release --prefix /path/to/prefix
cmake -S consumer -B consumer/build -DCMAKE_PREFIX_PATH=/path/to/prefix
```

The installed package provides `varpro::varpro`; the consumer must provide an
Eigen 3.4+ CMake package. For a source-tree consumer, use `add_subdirectory`
and link the same target directly.

For a single-configuration generator, add `-DCMAKE_BUILD_TYPE=Release` when
configuring. Alternatively, use a plain Eigen checkout without installing it:

```sh
git clone --depth 1 --branch 3.4.0 https://gitlab.com/libeigen/eigen.git build/_deps/eigen-src
cmake -S . -B build -DEIGEN3_INCLUDE_DIR=build/_deps/eigen-src
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

Run `build/Release/double_exponential.exe` with Visual Studio, or
`build/double_exponential` with a single-configuration generator. The example
fits two double-exponential curves and prints the shared time constants,
each curve's coefficients, and the squared residual error.

## Use

```cpp
#include <varpro/varpro.hpp>
#include <cmath>

varpro::Vector t = varpro::Vector::LinSpaced(50, 0.0, 10.0);
varpro::Problem problem;
problem.observations.resize(t.size(), 1);
for (Eigen::Index i = 0; i < t.size(); ++i)
    problem.observations(i, 0) = 3.0 * std::exp(-t[i] / 2.0) + 0.5;

problem.basis = [t](const varpro::Vector& alpha) -> varpro::Matrix {
    varpro::Matrix phi(t.size(), 2);
    phi.col(0) = (-t.array() / alpha[0]).exp().matrix();
    phi.col(1).setOnes();
    return phi;
};
problem.derivative = [t](const varpro::Vector& alpha, Eigen::Index) -> varpro::Matrix {
    varpro::Matrix dphi = varpro::Matrix::Zero(t.size(), 2);
    dphi.col(0) = ((-t.array() / alpha[0]).exp()
                  * t.array() / (alpha[0] * alpha[0])).matrix();
    return dphi;
};

varpro::Vector initial(1);
initial << 1.5;
auto result = varpro::fit(problem, initial);
// Check result.converged() before using this as a successful fit.
// result.parameters                  -> approximately [2]
// result.evaluation.coefficients      -> approximately [3, 0.5]^T
// result.evaluation.residuals         -> weighted residual matrix
```

Link the CMake target `varpro::varpro`, or compile `cpp/varpro.cpp` with
`include/` and the Eigen header directory on your include path. Consumers can
disable `VARPRO_BUILD_EXAMPLE` and `BUILD_TESTING`.

`evaluate(problem, alpha)` exposes the linear solution, residuals, approximate
Jacobian, and numerical rank at a fixed parameter vector. This also allows use
with another nonlinear optimizer without changing the VarPro calculation.

## Numerical contract

- Matrices use real `double` values. Rows are samples and columns are datasets.
  The basis and its analytical derivatives must have the same fixed shape.
- Optional `problem.weights` contains shared, nonnegative residual multipliers.
  For standard deviations `sigma`, use `1/sigma`. Empty means unit weights.
- The Jacobian uses the Kaufman approximation, as the Rust implementation does.
  It is generally not the exact residual derivative away from a perfect fit.
- Rank-deficient linear problems use a truncated SVD and minimum-norm solution.
  The same retained singular vectors form the Jacobian projector. This follows
  the article's range projector and corrects a difference in the checked-out
  Rust code, which uses all thin-U columns in that projector.
- Invalid inputs throw `std::invalid_argument`; nonfinite model evaluations or
  numerical outputs throw `std::domain_error`. Callback exceptions propagate.
  Invalid trial evaluations abort the fit. Parameter bounds are not provided;
  positive quantities can be represented by their logarithms in callbacks.
- `Options` sets the residual evaluation budget and LM `ftol`, `xtol`, `gtol`.
  `Status` distinguishes convergence, evaluation limit, stalled, and numerical
  failure. Convergence is a local stopping criterion, not proof of uniqueness.

See [CPP_DESIGN.md](CPP_DESIGN.md) for shapes, formulas, rank cutoff, validation,
and [verification evidence](CPP_DESIGN.md#verification).

## Why Eigen stays

Eigen supplies both SVD and the existing LM implementation through its bundled
`unsupported/Eigen/LevenbergMarquardt` module. Replacing those would add numerical
code to maintain and verify. There is no evidence here that removing Eigen
would improve runtime. This implementation prioritizes a small amount of
readable application code; no speedup over Rust is claimed.

## References and original implementation

- [Variable projection fundamentals](https://geo-ant.github.io/blog/2020/variable-projection-part-1-fundamentals/)
- [Global fitting of multiple right-hand sides](https://geo-ant.github.io/blog/2024/variable-projection-part-2-multiple-right-hand-sides/)
- [Original repository](https://github.com/n01r1r/varpro-cpp) at reference commit
  `29d847047966c0dd93550d7c5222bdfa34befdea`.
- [Archived original README](docs/original/README.md), retained verbatim for
  provenance and the original citations.
- [Port references and provenance](docs/REFERENCES.md).

The VarPro equations and weighting follow those references. The original Rust
implementation and its Rust/MATLAB/Python support files were removed from the
maintained tree; the archived README and links above preserve provenance.
Eigen's nonlinear optimizer differs from the original `levenberg-marquardt`,
so iteration counts and optimization trajectories need not match. The original
MIT license and attribution are retained in [LICENSE](LICENSE).
