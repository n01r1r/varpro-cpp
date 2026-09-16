# C++ implementation contract

## Purpose and sources

A small, usable, real double-precision C++ VarPro implementation. The original
Rust code was the algorithmic reference for this port; its API, builder
framework, fit statistics, and optional LAPACK backends are intentionally not
carried into the maintained C++ tree.

Reference checkout: `29d847047966c0dd93550d7c5222bdfa34befdea`.

- [Part 1: fundamentals](https://geo-ant.github.io/blog/2020/variable-projection-part-1-fundamentals/)
- [Part 2: multiple right-hand sides](https://geo-ant.github.io/blog/2024/variable-projection-part-2-multiple-right-hand-sides/)
- Archived original README: `docs/original/README.md`.
- Port provenance and direct references: `docs/REFERENCES.md`.

## Decisions

C++17, Eigen only. Eigen supplies SVD and its existing
`unsupported/Eigen/LevenbergMarquardt` solver (a module bundled with Eigen, not
another dependency). This keeps matrix decompositions and optimizer internals
out of the implementation. No dependency-free backend or speedup claim.

Use a declaration header plus one implementation file. Public API consists of
matrix callbacks, `evaluate`, and `fit`. Do not add a generic backend layer.

## Shapes, units, and objective

- `parameters`: q > 0 nonlinear parameters. Coordinates and physical units belong
  to the user's callbacks; the library does not rescale or reinterpret units.
- `observations`: m x s, m > 0 and s > 0; each column is one dataset.
- `basis(parameters)`: m x n, 0 < n <= m, invariant shape throughout a fit.
- `derivative(parameters, k)`: m x n partial derivative of the basis with respect
  to parameter k, 0 <= k < q. Its units are basis units / parameter-k units.
- `weights`: empty (identity), or m finite nonnegative residual multipliers with
  at least one positive entry. All datasets share the basis and weights. These
  are NOT squared-error multipliers: pass 1/sigma, not 1/sigma^2.
- `coefficients`: n x s, unconstrained real linear coefficients.
- `residuals`: m x s, W * (Y - Phi * C), in weighted observation units.
- `jacobian`: (m*s) x q, stacked by dataset/column, Kaufman approximation.
- Objective: sum of all squared weighted residual entries, without averaging.
  `fit` additionally needs m*s >= q for Eigen's LM.

For A = W*Phi and B = W*Y, compute one thin SVD per parameter evaluation.
Retain singular values strictly greater than
`sigma_max * max(m,n) * epsilon(double)`, matching Rust's coefficient cutoff.
Use retained U_r, V_r, sigma_r to compute minimum-norm C = A^+ B and R = B-A*C.
For D_k = W*dPhi/dalpha_k:

    J[:,k] = vec(U_r * (U_r^T * (D_k*C)) - D_k*C).

Never build an m x m projector, explicit pseudoinverse, or a block matrix with
one basis per dataset. Reuse the SVD between residual and Jacobian callbacks.
Reassociate `(U_r*(U_r^T*D_k)-D_k)*C` when s > q, as the Rust code does.

The Jacobian omits Part 1's b_k term, as the Rust implementation does. At nonzero
residual it is generally NOT the exact residual derivative; compare objective
gradients there, and compare residual finite differences at zero residual.

### Rank clarification

Rust thresholds singular values when solving C but uses all thin U columns in
the Jacobian projector. For a deficient basis those columns can include null
directions. C++ follows Part 1's range projector by using only retained columns
consistently. Full-rank formulas agree. Derivatives at rank transitions need not
exist. Rank zero is allowed for evaluation and produces C=0 and J=0; it does not
prove model identifiability or a useful optimum.

## Invalid cases and termination

Reject missing callbacks, invalid shapes, nonfinite parameters/data/weights,
negative/all-zero weights, and invalid options with `std::invalid_argument`.
Reject nonfinite model values, derivatives, linear solutions, residuals, Jacobian,
or overflowed squared error with `std::domain_error`. Callback exceptions
propagate. An invalid trial aborts the fit; it is never silently discarded or
reported as convergence. Users needing positive parameters can supply a log
parameterization in their callbacks. No bounds API is included.

Expose success, evaluation-limit termination, and stalled/numerical failure
separately. Returned parameters and coefficients/residuals must refer to the
same final accepted point, including a fit that exhausts its evaluation budget.
Eigen LM is not the Rust `levenberg-marquardt` implementation; trajectory,
iteration counts, and bitwise results need not match.

## Acceptance checks

Build with CMake and run CTest in Release. Deterministic cases must exercise:
single and multiple RHS recovery; weighted fit and manual preweighting
equivalence; residual orthogonality; Jacobian zero-residual finite differences
and nonzero-residual objective gradient; rank-deficient minimum-norm solve and
range projector; rank zero; invalid inputs/model outputs; evaluation-limit
non-success and final-point consistency. Assertions must remain active in
Release. Run the documented example and inspect its actual output.

Record actual compiler/Eigen versions, commands, and results after verification.
Tests verify implementation contracts, not scientific validation or speed.

## Verification

Environment: Windows x64, MSVC 19.44.35223.0 (Visual Studio 2022),
CMake 4.2.0-rc1, Eigen 3.4.0 at
`3147391d946bb4b6c68edd901f2add6ac1f31f8c`.

The Codex shell supplied duplicate `Path`/`PATH` environment entries, which
MSBuild rejected before compiling any project code. These process-local
PowerShell lines normalized them for configuration and build:

```powershell
$varproBuildPath = $env:PATH
Remove-Item Env:PATH
$env:Path = $varproBuildPath
cmake -S . -B build-msvc -G 'Visual Studio 17 2022' -A x64 -DEIGEN3_INCLUDE_DIR=C:/workspace/varpro-cpp/build/_deps/eigen-src
cmake --build build-msvc --config Release --parallel 2
ctest --test-dir build-msvc -C Release --output-on-failure
.\build-msvc\Release\double_exponential.exe
```

Verified on 2026-09-16: Release library, test runner, and example built
successfully. CTest passed 1/1 executable, containing six passing groups:

- Single RHS and three-RHS recovery.
- Weighted residual orthogonality, manual preweight equivalence of evaluations
  and completed fits.
- Zero-residual finite differences for single and three-RHS Jacobians, plus a
  nonzero-residual objective-gradient check.
- Rank-deficient minimum-norm solve/range projector and rank-zero behavior.
- Finite extreme-scale SVD cutoff (`sigma_max = 1e308`).
- Invalid inputs/evaluations and a strict one-call budget at a nonstationary
  initial point, with consistent final parameters, coefficients, and residuals.

The example exited successfully with:

```text
status: converged
parameters: 1 3
coefficients:
   4  1.5
 2.5   -2
   1 0.25
squared error: 2.60254765245e-29
```

The original README is preserved byte-for-byte as `docs/original/README.md`.
Rust source and Cargo metadata were removed from the maintained tree; no Rust
test suite remains to run. No C++/Rust runtime comparison or performance
benchmark was run.

## Complexity audit and cleanup

On 2026-09-16, Luna performed the requested `ponytail-audit` across the owned
repository, excluding Git internals, generated builds, and downloaded Eigen.
The original README and citation links remain under `docs/`; Rust, MATLAB,
Python, benchmark, and Rust test assets were removed. Eigen remains the only
C++ dependency.

Accepted cleanup preserves the numerical contract and public API:

- Remove `evaluate`'s duplicate input validation; its immediate `prepare` call
  performs the same validation.
- Remove the final whole-Jacobian finite scan; every column matrix is already
  checked before its entries are copied into the output. All input, model,
  intermediate, and per-column finite checks remain.

Removed redundant generated directories `build-msvc2` and `build-ninja`, plus
the failed `build/CMakeFiles` and `build/CMakeCache.txt`. Their resolved paths
were checked to stay inside this repository and contained no filesystem links.
Removed file sizes total 153,086,909 bytes (about 146 MiB) in the first cleanup;
this is a filesystem cleanup count, not a runtime measurement. The later
package verification trees (`build-msvc`, `build-verify`, `consumer-smoke`)
and the temporary local Eigen checkout were removed after their checks, so no
generated build output is kept in the source tree.

After applying both findings, `cpp/varpro.cpp` decreased from 318 to 315 lines;
no dependencies were removed. Rebuilt MSVC Release successfully and reran all
six CTest groups (1/1 executable passed). The example exited with code 0 and
printed the same parameters, coefficients, and squared error shown above.
The public header and existing tests were unchanged by this cleanup.

The installable package was also checked on 2026-09-16. A fresh source build
using the installed Eigen 3.4.1 CMake package was configured, built, and tested.
After `cmake --install build-verify --config Release --prefix
build-verify/install`, a separate CMake consumer configured with
`find_package(varpro 0.1 CONFIG REQUIRED)` and linked `varpro::varpro`. It
compiled and ran successfully with MSVC 19.44. The installed export contains
`${_IMPORT_PREFIX}/include` and the external `Eigen3::Eigen` target; it contains
no checkout-specific Eigen path.
