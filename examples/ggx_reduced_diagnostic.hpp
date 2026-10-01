#ifndef EXAMPLES_GGX_REDUCED_DIAGNOSTIC_HPP_
#define EXAMPLES_GGX_REDUCED_DIAGNOSTIC_HPP_

// Example-only affine-offset adapter, not a bounded VarPro API or GGX fitter.
#include <varpro/varpro.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include <Eigen/Geometry>
#include <Eigen/SVD>

namespace ggx_example {
using varpro::Matrix;
using varpro::Vector;
constexpr double kPi = 3.14159265358979323846;

struct Geometry {
  Vector a;
  Matrix half_local;  // Halfway vectors in a fixed tangent/normal frame.
  Vector fresnel;
  double normal_dot_view;
};

// Calibrated distant unit lights; fixed unit normal/view; all dots positive.
inline Geometry make_geometry(const Matrix& lights, const Eigen::Vector3d& n,
                              const Eigen::Vector3d& v) {
  if (lights.rows() <= 0 || lights.cols() != 3 || !lights.allFinite() ||
      !n.allFinite() || !v.allFinite() || std::abs(n.norm() - 1.0) > 1e-12 ||
      std::abs(v.norm() - 1.0) > 1e-12 || n.dot(v) <= 0.0)
    throw std::invalid_argument("invalid fixed GGX geometry");
  const Eigen::Vector3d u = n.unitOrthogonal(), b = n.cross(u);
  Geometry g{Vector(lights.rows()), Matrix(lights.rows(), 3),
             Vector(lights.rows()), n.dot(v)};
  for (Eigen::Index i = 0; i < lights.rows(); ++i) {
    const Eigen::Vector3d l = lights.row(i).transpose();
    if (std::abs(l.norm() - 1.0) > 1e-12 || n.dot(l) <= 0.0)
      throw std::invalid_argument("lights must be unit and front-facing");
    const Eigen::Vector3d h = (l + v).normalized();
    g.a(i) = n.dot(l) / kPi;
    g.half_local.row(i) << u.dot(h), b.dot(h), n.dot(h);
    g.fresnel(i) = 0.04 + 0.96 * std::pow(1.0 - v.dot(h), 5);
  }
  return g;
}

inline Geometry fixture_geometry() {
  Matrix lights(24, 3);
  for (int ring = 0; ring < 3; ++ring) {
    const double polar = (20.0 + 25.0 * ring) * kPi / 180.0;
    for (int j = 0; j < 8; ++j) {
      const double azimuth = 2.0 * kPi * j / 8.0;
      lights.row(8 * ring + j) << std::sin(polar) * std::cos(azimuth),
          std::sin(polar) * std::sin(azimuth), std::cos(polar);
    }
  }
  return make_geometry(lights, Eigen::Vector3d::UnitZ(),
                       Eigen::Vector3d::UnitZ());
}

struct Specular {
  Vector s;
  Matrix derivative;
};

// theta = [log(alpha_t), log(alpha_b), tangent angle in radians].
// Physical alpha in (0,1]; G=1 and fixed Schlick F0=0.04.
inline Specular specular(const Geometry& g, const Vector& theta) {
  if (theta.size() != 3 || !theta.allFinite() || theta(0) > 0.0 ||
      theta(1) > 0.0)
    throw std::invalid_argument("expected physical log roughness and angle");
  const double at = std::exp(theta(0)), ab = std::exp(theta(1));
  const double it = 1.0 / (at * at), ib = 1.0 / (ab * ab);
  const double c = std::cos(theta(2)), s = std::sin(theta(2));
  Specular result{Vector(g.a.size()), Matrix(g.a.size(), 3)};
  for (Eigen::Index i = 0; i < g.a.size(); ++i) {
    const double x = c * g.half_local(i, 0) + s * g.half_local(i, 1);
    const double y = -s * g.half_local(i, 0) + c * g.half_local(i, 1);
    const double z = g.half_local(i, 2);
    const double q = x * x * it + y * y * ib + z * z;
    const double value = g.fresnel(i) /
        (4.0 * kPi * g.normal_dot_view * at * ab * q * q);
    result.s(i) = value;
    result.derivative.row(i) << value * (-1.0 + 4.0 * x * x * it / q),
        value * (-1.0 + 4.0 * y * y * ib / q),
        -4.0 * value * x * y * (it - ib) / q;
  }
  if (!result.s.allFinite() || !result.derivative.allFinite())
    throw std::domain_error("nonfinite GGX evaluation");
  return result;
}

struct Spectrum {
  Eigen::Index rank = 0;
  varpro::Diagnostics diagnostics;
};

// Separate SVD of J_red, with the README's strict relative-cutoff convention.
// The public Evaluation diagnostics describe the linear basis, not J_red.
inline Spectrum spectrum(const Matrix& jacobian, double rcond = -1.0) {
  if (jacobian.rows() <= 0 || jacobian.cols() <= 0 ||
      !jacobian.allFinite() || !std::isfinite(rcond))
    throw std::invalid_argument("invalid Jacobian spectrum input");
  Eigen::JacobiSVD<Matrix> svd(jacobian);
  Spectrum result;
  auto& d = result.diagnostics;
  d.singular_values = svd.singularValues();
  d.sigma_max = d.singular_values(0);
  const double cutoff = rcond < 0.0
      ? static_cast<double>((std::max)(jacobian.rows(), jacobian.cols())) *
            std::numeric_limits<double>::epsilon()
      : rcond;
  for (Eigen::Index i = 0; i < d.singular_values.size(); ++i) {
    if (d.sigma_max > 0.0 && d.singular_values(i) / d.sigma_max > cutoff) {
      ++result.rank;
      d.sigma_min_retained = d.singular_values(i);
    }
  }
  if (result.rank > 0)
    d.condition_estimate = d.sigma_max / d.sigma_min_retained;
  return result;
}

enum class MeanBranch { interior, lower, upper };
struct Reduced {
  double mu_free;
  double mu;
  MeanBranch branch;
  bool at_switch;
  Vector residual;
  Matrix jacobian;
  Eigen::Index linear_rank;
  varpro::Diagnostics linear_diagnostics;
};

// Fixed n, weights and [lower,upper]. No derivative of stage-one estimates.
// At a switch the returned J is a branch derivative, not a two-sided one.
inline Reduced evaluate(const Geometry& g, const Vector& observations,
                        const Vector& weights, const Vector& theta,
                        double lower, double upper) {
  if (observations.size() != g.a.size() || !std::isfinite(lower) ||
      !std::isfinite(upper) || lower > upper)
    throw std::invalid_argument("invalid neutral observations or mean bounds");
  const auto model = specular(g, theta);
  varpro::Problem p;
  p.observations.resize(g.a.size(), 4);
  p.observations.col(0) = observations - model.s;
  p.observations.rightCols(3) = -model.derivative;
  p.weights = weights;
  p.basis = [&g](const Vector&) { return Matrix(g.a); };
  p.derivative = [&g](const Vector&, Eigen::Index) {
    return Matrix::Zero(g.a.size(), 1);
  };
  // Four RHS reuse one SVD: physical mean plus three derivative projections.
  // The library Jacobian is zero because these RHS are fixed within this call.
  // They change between calls: never pass this construction to varpro::fit.
  const auto e = varpro::evaluate(p, theta,
                                 varpro::JacobianMode::retained_subspace);
  if (e.rank != 1)
    throw std::domain_error("diffuse mean has no retained observation direction");
  const double free = e.coefficients(0, 0);
  const double mu = (std::max)(lower, (std::min)(upper, free));
  const MeanBranch branch = free <= lower ? MeanBranch::lower
      : free >= upper ? MeanBranch::upper : MeanBranch::interior;
  const double tolerance = 64.0 * std::numeric_limits<double>::epsilon() *
      (std::max)({1.0, std::abs(free), std::abs(lower), std::abs(upper)});
  const bool at_switch = lower < upper &&
      (std::abs(free - lower) <= tolerance || std::abs(free - upper) <= tolerance);
  Vector residual = observations - model.s - g.a * mu;
  Matrix jacobian = branch == MeanBranch::interior
      ? Matrix(e.residuals.rightCols(3)) : Matrix(-model.derivative);
  if (weights.size() != 0) {
    residual.array() *= weights.array();
    if (branch != MeanBranch::interior)
      for (Eigen::Index i = 0; i < weights.size(); ++i)
        jacobian.row(i) *= weights(i);
  }
  return {free, mu, branch, at_switch, residual, jacobian, e.rank, e.diagnostics};
}
}  // namespace ggx_example
#endif  // EXAMPLES_GGX_REDUCED_DIAGNOSTIC_HPP_
