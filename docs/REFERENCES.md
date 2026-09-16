# VarPro C++ port: provenance and references

This repository is a small C++17 port of the variable projection workflow from
the original repository:

- Repository: <https://github.com/n01r1r/varpro-cpp>
- Reference checkout used for the port: `29d847047966c0dd93550d7c5222bdfa34befdea`
- Original README, retained verbatim: [original/README.md](original/README.md)

The implementation follows the mathematical presentation in:

- Geo's Notepad, [Variable Projection Method: Nonlinear Least Squares Fitting
  with VarPro](https://geo-ant.github.io/blog/2020/variable-projection-part-1-fundamentals/)
- Geo's Notepad, [Global Fitting of Multiple Right Hand Sides with Variable
  Projection](https://geo-ant.github.io/blog/2024/variable-projection-part-2-multiple-right-hand-sides/)
- Golub and Pereyra, [The Differentiation of Pseudo-Inverses and Nonlinear
  Least Squares Problems Whose Variables Separate](https://doi.org/10.1137/0710036)
- O'Leary and Rust, [Variable projection for nonlinear least squares
  problems](https://doi.org/10.1007/s10589-012-9492-9)

The archived README contains the original crate's additional citations and
acknowledgements. The maintained implementation uses Eigen's SVD and bundled
Levenberg-Marquardt module, while exposing a deliberately smaller C++ API.
The Rust, MATLAB, Python, benchmark, and Rust-specific test files from the
reference checkout are not part of this maintained C++ codebase.
