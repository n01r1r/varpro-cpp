#include "../../examples/ggx_reduced_diagnostic.hpp"

#include <iostream>
#include <string>

namespace {
using namespace ggx_example;

void check(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

Vector parameters(double at = 0.30, double ab = 0.50) {
  Vector x(3);
  x << std::log(at), std::log(ab), 0.4;
  return x;
}

void test_affine_projection_and_finite_differences() {
  const auto g = fixture_geometry();
  const Vector x = parameters();
  Vector y = 0.4 * g.a + specular(g, parameters(0.25, 0.55)).s;
  Vector w(g.a.size());
  for (Eigen::Index i = 0; i < y.size(); ++i) {
    y(i) += 1e-3 * std::sin(1.7 * static_cast<double>(i));
    w(i) = 1.0 + 0.1 * (i % 3);
  }
  const auto model = specular(g, x);
  const auto e = evaluate(g, y, w, x, 0.2, 0.8);
  const Vector a = g.a.cwiseProduct(w);
  const Vector b = (y - model.s).cwiseProduct(w);
  const double manual_mu = a.dot(b) / a.squaredNorm();
  check(e.branch == MeanBranch::interior && !e.at_switch,
        "nonzero-residual fixture must be strictly interior");
  check(std::abs(e.mu - manual_mu) < 1e-12, "weighted conditional mean");
  check(e.residual.norm() > 1e-3, "FD must exercise a nonzero residual");
  check(std::abs(a.dot(e.residual)) < 1e-12, "mean residual orthogonality");
  check(e.linear_rank == 1 &&
        std::abs(e.linear_diagnostics.condition_estimate - 1.0) < 1e-12,
        "one-column linear condition");
  Matrix full(g.a.size(), 4);
  full.col(0) = -a;
  full.rightCols(3) = -model.derivative;
  for (Eigen::Index i = 0; i < w.size(); ++i)
    full.row(i).tail(3) *= w(i);
  const Matrix information = full.transpose() * full;
  const Matrix conditional = information.bottomRightCorner(3, 3) -
      information.bottomLeftCorner(3, 1) *
      information.topRightCorner(1, 3) / information(0, 0);
  check((conditional - e.jacobian.transpose() * e.jacobian).norm() < 1e-12,
        "conditional Jacobian agrees with mean-eliminated information");
  for (Eigen::Index k = 0; k < 3; ++k) {
    for (double h : {1e-4, 1e-5, 1e-6}) {
      Vector plus = x, minus = x;
      plus(k) += h;
      minus(k) -= h;
      const auto ep = evaluate(g, y, w, plus, 0.2, 0.8);
      const auto em = evaluate(g, y, w, minus, 0.2, 0.8);
      check(ep.branch == e.branch && em.branch == e.branch,
            "FD stencil crossed a mean branch");
      const Vector fd = (ep.residual - em.residual) / (2.0 * h);
      check((fd - e.jacobian.col(k)).norm() < 2e-7 * (1.0 + fd.norm()),
            "affine reduced-residual finite difference");
      const Vector ds = (specular(g, plus).s - specular(g, minus).s) / (2.0 * h);
      check((ds - model.derivative.col(k)).norm() < 2e-7 * (1.0 + ds.norm()),
            "GGX basis derivative including log chain rule");
      const double gradient_fd =
          (ep.residual.squaredNorm() - em.residual.squaredNorm()) / (2.0 * h);
      check(std::abs(gradient_fd - 2.0 * e.jacobian.col(k).dot(e.residual)) < 1e-8,
            "reduced objective gradient");
    }
  }
  // Manual preweighting equivalence of the four shared-SVD RHS projections.
  Matrix raw(g.a.size(), 4);
  raw.col(0) = y - model.s;
  raw.rightCols(3) = -model.derivative;
  for (Eigen::Index i = 0; i < raw.rows(); ++i) raw.row(i) *= w(i);
  varpro::Problem p;
  p.observations = raw;
  p.basis = [a](const Vector&) { return Matrix(a); };
  p.derivative = [a](const Vector&, Eigen::Index) {
    return Matrix::Zero(a.size(), 1);
  };
  const auto preweighted = varpro::evaluate(p, x);
  check((preweighted.residuals.col(0) - e.residual).norm() < 1e-12 &&
        (preweighted.residuals.rightCols(3) - e.jacobian).norm() < 1e-12,
        "shared-SVD MRHS projection / preweighting equivalence");
}

void test_free_gain_counterexample() {
  const auto g = fixture_geometry();
  const Vector x = parameters();
  const Vector s = specular(g, x).s;
  const Vector y = 0.4 * g.a + 2.0 * s;
  varpro::Problem wrong;
  wrong.observations = y;
  wrong.basis = [g, s](const Vector&) {
    Matrix phi(g.a.size(), 2);
    phi << g.a, s;
    return phi;
  };
  wrong.derivative = [g](const Vector&, Eigen::Index) {
    return Matrix::Zero(g.a.size(), 2);
  };
  const auto free_gain = varpro::evaluate(wrong, x);
  check(free_gain.rank == 2 && free_gain.residuals.norm() < 1e-12 &&
        std::abs(free_gain.coefficients(1, 0) - 2.0) < 1e-12,
        "free specular coefficient fits an unintended gain");
  const auto physical = evaluate(g, y, Vector(), x, 0.2, 0.8);
  check(physical.residual.norm() > 1e-3,
        "fixed-gain physical GGX differs from [a,s] VarPro model");
}

void test_bounds_and_switch() {
  const auto g = fixture_geometry();
  const Vector x = parameters(), w = Vector::Ones(g.a.size());
  const Vector s = specular(g, x).s;
  for (double mean : {0.1, 0.9}) {
    const Vector y = s + mean * g.a;
    const auto e = evaluate(g, y, w, x, 0.2, 0.8);
    check(e.branch == (mean < 0.2 ? MeanBranch::lower : MeanBranch::upper),
          "active mean bound");
    check(std::abs(e.mu - (mean < 0.2 ? 0.2 : 0.8)) < 1e-12,
          "clipped conditional mean");
    check((e.jacobian + specular(g, x).derivative).norm() < 1e-12,
          "active-bound derivative must not project out diffuse direction");
    for (Eigen::Index k = 0; k < 3; ++k) {
      Vector plus = x, minus = x;
      plus(k) += 1e-6;
      minus(k) -= 1e-6;
      const auto ep = evaluate(g, y, w, plus, 0.2, 0.8);
      const auto em = evaluate(g, y, w, minus, 0.2, 0.8);
      check(ep.branch == e.branch && em.branch == e.branch,
            "active-bound FD stencil");
      check(((ep.residual - em.residual) / 2e-6 - e.jacobian.col(k)).norm() < 1e-8,
            "active-bound finite difference");
    }
  }
  const Vector y = s + 0.4 * g.a;
  // Use exactly the computed free mean as the boundary to avoid roundoff
  // deciding which side of a nominal 0.4 the base point lies on.
  const double lower = evaluate(g, y, w, x, 0.0, 1.0).mu_free;
  const auto e = evaluate(g, y, w, x, lower, 0.8);
  check(e.at_switch, "must mark the active-set switch");
  Vector plus = x, minus = x;
  plus(0) += 1e-6;
  minus(0) -= 1e-6;
  const auto ep = evaluate(g, y, w, plus, lower, 0.8);
  const auto em = evaluate(g, y, w, minus, lower, 0.8);
  check(ep.branch != em.branch, "switch stencil must straddle branches");
  const Vector right = (ep.residual - e.residual) / 1e-6;
  const Vector left = (e.residual - em.residual) / 1e-6;
  check((right - left).norm() > 1e-3,
        "branch switch has no common two-sided residual derivative");
}

void test_local_rank_change() {
  const auto g = fixture_geometry();
  const Vector w = Vector::Ones(g.a.size());
  const Vector y = 0.4 * g.a + specular(g, parameters(0.4, 0.4)).s;
  const auto iso = evaluate(g, y, w, parameters(0.4, 0.4), 0.2, 0.8);
  const auto anis = evaluate(g, y, w, parameters(0.3, 0.5), 0.2, 0.8);
  check(spectrum(iso.jacobian, 1e-10).rank == 2 &&
        spectrum(anis.jacobian, 1e-10).rank == 3, "isotropy local rank change");
  check(iso.jacobian.col(2).norm() == 0.0, "isotropic angle is unobservable");
  check(iso.linear_rank == 1 && anis.linear_rank == 1,
        "linear mean rank does not diagnose material rank");
  const auto d = spectrum(iso.jacobian, 1e-10).diagnostics;
  check(d.singular_values(2) == 0.0 && d.condition_estimate > 0.0,
        "retained condition excludes the missing angle direction");
  const auto near = evaluate(g, y, w, parameters(0.4, 0.400001), 0.2, 0.8);
  check(spectrum(near.jacobian, 1e-10).rank == 3 &&
        spectrum(near.jacobian, 1e-3).rank == 2,
        "reported numerical rank depends on the explicit diagnostic cutoff");
  check(spectrum(Matrix::Zero(24, 3)).rank == 0 &&
        spectrum(Matrix::Zero(24, 3)).diagnostics.condition_estimate == 0.0,
        "rank-zero diagnostic sentinel");
}

void test_linear_rank_cutoff_crossing() {
  // This is a generic linear-basis contract test, separate from smooth GGX
  // isotropy. An FD stencil crossing a cutoff is not a derivative check.
  varpro::Problem p;
  p.observations = Matrix::Zero(3, 1);
  p.observations(1, 0) = 1.0;
  p.basis = [](const Vector& x) {
    Matrix a = Matrix::Zero(3, 2);
    a(0, 0) = 1.0;
    a(1, 1) = x(0);
    return a;
  };
  p.derivative = [](const Vector&, Eigen::Index) {
    Matrix d = Matrix::Zero(3, 2);
    d(1, 1) = 1.0;
    return d;
  };
  varpro::LinearOptions options;
  options.rcond = 0.01;
  Vector x(1), plus(1), minus(1);
  x << 0.01;
  plus << 0.010001;
  minus << 0.009999;
  const auto e = varpro::evaluate(p, x, options,
                               varpro::JacobianMode::retained_subspace);
  const auto ep = varpro::evaluate(p, plus, options);
  const auto em = varpro::evaluate(p, minus, options);
  check(e.rank == 1 && ep.rank == 2 && em.rank == 1,
        "strict cutoff rank crossing");
  check((ep.residuals - em.residuals).norm() > 0.9 &&
        e.jacobian.norm() == 0.0,
        "cutoff residual jump cannot be validated as a smooth derivative");
}
}  // namespace

int main() {
  try {
    test_affine_projection_and_finite_differences();
    test_bounds_and_switch();
    test_local_rank_change();
    test_free_gain_counterexample();
    test_linear_rank_cutoff_crossing();
    std::cout << "[PASS] GGX affine FD, bounds/switch, local rank, cutoff crossing\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "[FAIL] " << error.what() << '\n';
    return 1;
  }
}
