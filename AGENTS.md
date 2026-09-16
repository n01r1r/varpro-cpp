# Project objective

Build the smallest readable C++ implementation of the variable projection
algorithm described in Geo's two articles and implemented by the original
repository's Rust crate. The C++ port is the only maintained code; preserve
the original README and source links as documentation under `docs/`.

- User-facing explanations are Korean; follow the existing English code/docs style.
- Use Luna for substantial implementation, investigation, and test-writing tasks.
- Use Eigen as the only C++ dependency. A second dependency-free matrix backend is
  outside the current scope: do not expand the code to remove Eigen speculatively.
- Prefer explicit double-precision matrices and callbacks over model builders,
  scalar/backend template frameworks, statistics, or performance infrastructure.
- Include single and multiple right-hand sides with shared diagonal weights.
- Read CPP_DESIGN.md before changing the numerical contract. Keep scientific
  meaning, rank handling, weighting, and stopping outcomes explicit.
- Verify deterministic numerical invariants and end-to-end fits. Never claim a
  speedup or equivalence of optimizer trajectories without measurement.
