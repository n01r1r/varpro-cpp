#ifndef VARPRO_VARPRO_HPP_
#define VARPRO_VARPRO_HPP_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

/**
 * @file varpro.hpp
 * @brief Public API for variable projection of separable nonlinear
 *        least-squares problems.
 */

namespace varpro {

using Index = std::ptrdiff_t;

namespace detail {
inline std::size_t checked_size(Index count) {
  if (count < 0) throw std::invalid_argument("negative dimension");
  return static_cast<std::size_t>(count);
}
inline std::size_t checked_size(Index rows, Index cols) {
  if (rows < 0 || cols < 0 ||
      (cols && rows > (std::numeric_limits<Index>::max)() / cols))
    throw std::invalid_argument("invalid matrix dimensions");
  return static_cast<std::size_t>(rows * cols);
}
}  // namespace detail

/** Dynamic double vector with no external dependencies. */
class Vector {
 public:
  Vector() = default;
  explicit Vector(Index n) : v_(detail::checked_size(n), 0.0) {}
  Vector(std::initializer_list<double> v) : v_(v) {}
  Vector(const double* values, Index n)
      : v_(values, values + detail::checked_size(n)) {}
  static Vector Zero(Index n) { return Vector(n); }
  static Vector Constant(Index n, double v) { Vector r(n); std::fill(r.v_.begin(), r.v_.end(), v); return r; }
  static Vector LinSpaced(Index n, double first, double last) {
    Vector r(n);
    for (Index i = 0; i < n; ++i) r(i) = n == 1 ? first : first + (last - first) * i / (n - 1);
    return r;
  }
  Index size() const { return static_cast<Index>(v_.size()); }
  void resize(Index n) { v_.assign(detail::checked_size(n), 0.0); }
  double* data() { return v_.data(); }
  const double* data() const { return v_.data(); }
  double& operator()(Index i) { return v_[static_cast<std::size_t>(i)]; }
  double operator()(Index i) const { return v_[static_cast<std::size_t>(i)]; }
  double& operator[](Index i) { return (*this)(i); }
  double operator[](Index i) const { return (*this)(i); }
  bool allFinite() const { for (double v : v_) if (!std::isfinite(v)) return false; return true; }
  double squaredNorm() const { double s = 0; for (double v : v_) s += v * v; return s; }
  double norm() const { double s = 0; for (double v : v_) s = std::hypot(s, v); return s; }
  double sum() const { double s = 0; for (double v : v_) s += v; return s; }
  double dot(const Vector& b) const {
    if (size() != b.size()) throw std::invalid_argument("vector shape mismatch");
    double s = 0; for (Index i = 0; i < size(); ++i) s += (*this)(i) * b(i);
    return s;
  }
  double maxCoeff() const {
    if (v_.empty()) throw std::invalid_argument("empty vector");
    return *std::max_element(v_.begin(), v_.end());
  }
  Vector cwiseAbs() const { Vector r = *this; for (double& v : r.v_) v = std::abs(v); return r; }
  void normalize() {
    if (!allFinite()) throw std::domain_error("nonfinite vector cannot be normalized");
    double scale = 0.0;
    for (double v : v_) scale = (std::max)(scale, std::abs(v));
    if (scale == 0.0) throw std::domain_error("zero vector cannot be normalized");
    for (double& v : v_) v /= scale;
    const double n = norm();
    for (double& v : v_) v /= n;
  }
  Vector& operator+=(const Vector& b) { if (size() != b.size()) throw std::invalid_argument("vector shape mismatch"); for (Index i = 0; i < size(); ++i) (*this)(i) += b(i); return *this; }
  Vector& operator-=(const Vector& b) { if (size() != b.size()) throw std::invalid_argument("vector shape mismatch"); for (Index i = 0; i < size(); ++i) (*this)(i) -= b(i); return *this; }
  Vector& operator*=(double s) { for (double& v : v_) v *= s; return *this; }
  Vector& operator/=(double s) { for (double& v : v_) v /= s; return *this; }
  class Initializer {
   public:
    Initializer(Vector& v, double first) : v_(v) { append(first); }
    Initializer& operator,(double x) { append(x); return *this; }
   private:
    void append(double x) { if (i_ >= v_.size()) throw std::invalid_argument("too many values"); v_(i_++) = x; }
    Vector& v_; Index i_ = 0;
  };
  Initializer operator<<(double first) { return Initializer(*this, first); }
 private:
  std::vector<double> v_;
};
inline Vector operator+(Vector a, const Vector& b) { return a += b; }
inline Vector operator-(Vector a, const Vector& b) { return a -= b; }
inline Vector operator*(Vector a, double s) { return a *= s; }
inline Vector operator*(double s, Vector a) { return a *= s; }
inline Vector operator/(Vector a, double s) { return a /= s; }

/** Dynamic column-major double matrix; comma initialization is row-major. */
class Matrix {
 public:
  Matrix() = default;
  Matrix(Index rows, Index cols) : rows_(rows), cols_(cols), v_(detail::checked_size(rows, cols), 0.0) {}
  Matrix(const Vector& b) : Matrix(b.size(), 1) { for (Index i = 0; i < rows_; ++i) (*this)(i, 0) = b(i); }
  static Matrix Zero(Index m, Index n) { return Matrix(m, n); }
  static Matrix Constant(Index m, Index n, double v) { Matrix r(m, n); std::fill(r.v_.begin(), r.v_.end(), v); return r; }
  static Matrix Ones(Index m, Index n) { return Constant(m, n, 1.0); }
  static Matrix Identity(Index m, Index n) { Matrix r(m, n); for (Index i = 0; i < (std::min)(m, n); ++i) r(i,i) = 1.; return r; }
  Index rows() const { return rows_; }
  Index cols() const { return cols_; }
  Index size() const { return static_cast<Index>(v_.size()); }
  void resize(Index m, Index n) { v_.assign(detail::checked_size(m, n), 0.0); rows_ = m; cols_ = n; }
  double* data() { return v_.data(); }
  const double* data() const { return v_.data(); }
  double& operator()(Index i, Index j) { return v_[static_cast<std::size_t>(j * rows_ + i)]; }
  double operator()(Index i, Index j) const { return v_[static_cast<std::size_t>(j * rows_ + i)]; }
  bool allFinite() const { for (double v : v_) if (!std::isfinite(v)) return false; return true; }
  double squaredNorm() const { double s = 0; for (double v : v_) s += v * v; return s; }
  double norm() const { double s = 0; for (double v : v_) s = std::hypot(s, v); return s; }
  double maxCoeff() const { if (v_.empty()) throw std::invalid_argument("empty matrix"); return *std::max_element(v_.begin(), v_.end()); }
  Matrix cwiseAbs() const { Matrix r = *this; for (double& v : r.v_) v = std::abs(v); return r; }
  Matrix cwiseProduct(const Matrix& b) const {
    require_shape(b); Matrix r(rows_, cols_);
    for (Index i = 0; i < size(); ++i) r.v_[i] = v_[i] * b.v_[i];
    return r;
  }
  Matrix transpose() const { Matrix r(cols_, rows_); for (Index j = 0; j < cols_; ++j) for (Index i = 0; i < rows_; ++i) r(j,i) = (*this)(i,j); return r; }
  Vector col(Index j) const { Vector r(rows_); for (Index i = 0; i < rows_; ++i) r(i) = (*this)(i,j); return r; }
  Matrix leftCols(Index n) const {
    if (n < 0 || n > cols_) throw std::invalid_argument("bad column count");
    Matrix r(rows_, n); for (Index j = 0; j < n; ++j) for (Index i = 0; i < rows_; ++i) r(i,j) = (*this)(i,j); return r;
  }
  Matrix& operator+=(const Matrix& b) { require_shape(b); for (Index i = 0; i < size(); ++i) v_[i] += b.v_[i]; return *this; }
  Matrix& operator-=(const Matrix& b) { require_shape(b); for (Index i = 0; i < size(); ++i) v_[i] -= b.v_[i]; return *this; }
  Matrix& operator*=(double s) { for (double& v : v_) v *= s; return *this; }
  Matrix& operator/=(double s) { for (double& v : v_) v /= s; return *this; }
  class Initializer {
   public:
    Initializer(Matrix& m, double x) : m_(m) { append(x); }
    Initializer& operator,(double x) { append(x); return *this; }
   private:
    void append(double x) { if (i_ >= m_.size()) throw std::invalid_argument("too many values"); m_(i_ / m_.cols_, i_ % m_.cols_) = x; ++i_; }
    Matrix& m_; Index i_ = 0;
  };
  Initializer operator<<(double first) { return Initializer(*this, first); }
 private:
  void require_shape(const Matrix& b) const { if (rows_ != b.rows_ || cols_ != b.cols_) throw std::invalid_argument("matrix shape mismatch"); }
  Index rows_ = 0, cols_ = 0;
  std::vector<double> v_;
};
inline Matrix operator+(Matrix a, const Matrix& b) { return a += b; }
inline Matrix operator-(Matrix a, const Matrix& b) { return a -= b; }
inline Matrix operator*(Matrix a, double s) { return a *= s; }
inline Matrix operator*(double s, Matrix a) { return a *= s; }
inline Matrix operator/(Matrix a, double s) { return a /= s; }
inline Matrix operator*(const Matrix& a, const Matrix& b) {
  if (a.cols() != b.rows()) throw std::invalid_argument("matrix product shape mismatch");
  Matrix r(a.rows(), b.cols());
  for (Index j = 0; j < b.cols(); ++j)
    for (Index k = 0; k < a.cols(); ++k)
      for (Index i = 0; i < a.rows(); ++i) r(i,j) += a(i,k) * b(k,j);
  return r;
}
inline Vector operator*(const Matrix& a, const Vector& b) {
  if (a.cols() != b.size()) throw std::invalid_argument("matrix-vector shape mismatch");
  Vector r(a.rows());
  for (Index k = 0; k < a.cols(); ++k)
    for (Index i = 0; i < a.rows(); ++i) r(i) += a(i,k) * b(k);
  return r;
}

/**
 * @brief Defines the observations and model callbacks for a VarPro problem.
 *
 * Let @f$Y@f$ be the observation matrix with shape @f$m \times s@f$ and let
 * @f$\Phi(\alpha)@f$ be the basis matrix with shape @f$m \times n@f$.
 * Each observation column is a separate dataset. The basis and derivative
 * callbacks share the nonlinear parameter vector and must return a consistent
 * basis width throughout a call to @c fit().
 *
 * @par Dimension contract
 * - @c observations has shape @f$m \times s@f$ with @f$m > 0@f$ and @f$s >
 * 0@f$.
 * - @c basis returns an @f$m \times n@f$ matrix with @f$0 < n \le m@f$.
 * - @c derivative returns an @f$m \times n@f$ matrix for every valid parameter
 *   index.
 * - The nonlinear parameter vector has @f$q > 0@f$ entries, and the derivative
 *   callback is called with @c k in the range @f$0 \le k < q@f$.
 *
 * @par Weight contract
 *
 * If @c weights is empty, unit weights are used. Otherwise it must contain
 * @f$m@f$ finite, nonnegative residual multipliers, with at least one positive
 * entry. Weights are applied once to each residual row; pass @f$1/\sigma@f$,
 * not @f$1/\sigma^2@f$, when @f$\sigma@f$ contains observation standard
 * deviations.
 */
struct Problem {
  /**
   * @brief Observations, with one dataset per column.
   *
   * The shape is @f$m \times s@f$, where @f$m > 0@f$ and @f$s > 0@f$.
   */
  Matrix observations;

  /**
   * @brief Returns the basis matrix @f$\Phi(\alpha)@f$.
   *
   * @param parameters Nonlinear parameter vector @f$\alpha@f$.
   * @return An @f$m \times n@f$ basis matrix with @f$0 < n \le m@f$.
   * @note The returned basis width must remain fixed during @ref fit.
   */
  std::function<Matrix(const Vector&)> basis;

  /**
   * @brief Returns a partial derivative of the basis matrix.
   *
   * @param parameters Nonlinear parameter vector @f$\alpha@f$.
   * @param k Zero-based nonlinear parameter index.
   * @return The @f$m \times n@f$ matrix
   *         @f$\partial\Phi(\alpha)/\partial\alpha_k@f$.
   */
  std::function<Matrix(const Vector&, Index)> derivative;

  /**
   * @brief Optional shared residual multipliers.
   *
   * An empty vector selects unit weights. Otherwise this contains one
   * multiplier per observation row and is shared by all datasets.
   */
  Vector weights;
};

/**
 * @brief Residual-Jacobian formula used by @c evaluate() or @c fit().
 */
enum class JacobianMode {
  /**
   * @brief Use the compact-SVD Kaufman approximation.
   *
   * This is the default. It generally differs from the exact residual
   * derivative when the residual is nonzero.
   */
  kaufman,

  /**
   * @brief Include the response of the eliminated linear coefficients.
   *
   * This is the full VarPro formula for a locally constant retained rank.
   * If an explicit cutoff discards a nonzero singular direction, the
   * discarded direction is treated as null by the implementation.
   */
  retained_subspace,

  /**
   * @brief Compatibility alias for @c retained_subspace.
   *
   * Prefer @c retained_subspace in new code. This mode is not a promise of
   * the derivative of a residual map that re-truncates a rotating SVD.
   */
  exact = retained_subspace
};

/**
 * @brief Controls the numerical-rank cutoff of the linear SVD solve.
 */
struct LinearOptions {
  /**
   * @brief Relative singular-value cutoff.
   *
   * Singular values satisfying
   * @f$\sigma_i > \mathtt{rcond}\,\sigma_{\max}@f$ are retained when this
   * value is nonnegative. A negative value selects the automatic cutoff
   * @f$\max(m,n)\,\epsilon@f$.
   */
  double rcond = -1.0;
};

/**
 * @brief Singular-value diagnostics for one weighted basis evaluation.
 *
 * The singular values belong to @f$W\Phi(\alpha)@f$, before numerical-rank
 * truncation. The condition estimate uses only the retained subspace:
 * @f$\sigma_{\max}/\sigma_{\min,\mathrm{retained}}@f$. It is zero when the
 * retained rank is zero and does not describe discarded directions.
 */
struct Diagnostics {
  /** @brief All singular values of the weighted basis, descending. */
  Vector singular_values;
  /** @brief Largest singular value of the weighted basis. */
  double sigma_max = 0.0;
  /** @brief Smallest singular value retained by the numerical-rank cutoff. */
  double sigma_min_retained = 0.0;
  /** @brief Condition estimate of the retained singular subspace. */
  double condition_estimate = 0.0;
};

/**
 * @brief Results computed at one nonlinear parameter vector.
 */
struct Evaluation {
  /**
   * @brief Minimum-norm linear coefficients, with shape @f$n \times s@f$.
   */
  Matrix coefficients;

  /**
   * @brief Weighted residuals, with one dataset per column.
   *
   * This is @f$W(Y - \Phi(\alpha)C)@f$ and has shape @f$m \times s@f$.
   */
  Matrix residuals;

  /**
   * @brief Selected residual Jacobian, with shape @f$(m s) \times q@f$.
   *
   * Each column is formed by stacking the corresponding residual column
   * for each dataset in column-major order. The formula is selected by the
   * @c JacobianMode passed to the evaluation or fit operation.
   */
  Matrix jacobian;

  /**
   * @brief Numerical rank retained by the compact SVD solve.
   */
  Index rank = 0;

  /**
   * @brief Singular-value and conditioning information for the weighted basis.
   */
  Diagnostics diagnostics;

  /**
   * @brief Returns the sum of squared weighted residual entries.
   *
   * @return @f$\|W(Y - \Phi(\alpha)C)\|_F^2@f$.
   */
  double squared_error() const { return residuals.squaredNorm(); }
};

/**
 * @brief Controls the nonlinear least-squares fit.
 */
struct Options {
  /**
   * @brief Residual-Jacobian formula used by the optimizer.
   */
  JacobianMode jacobian_mode = JacobianMode::kaufman;

  /**
   * @brief Options for the linear compact-SVD solve.
   */
  LinearOptions linear_options;

  /**
   * @brief Maximum number of residual callback evaluations.
   *
   * The final evaluation used to populate @ref FitResult is not counted.
   * This value must be positive.
   */
  int max_evaluations = 1000;

  /**
   * @brief Levenberg--Marquardt relative reduction tolerance.
   *
   * Must be finite and nonnegative.
   */
  double ftol = 1e-12;

  /**
   * @brief Levenberg--Marquardt parameter-step tolerance.
   *
   * Must be finite and nonnegative.
   */
  double xtol = 1e-12;

  /**
   * @brief Levenberg--Marquardt gradient tolerance.
   *
   * Must be finite and nonnegative.
   */
  double gtol = 1e-12;
};

/**
 * @brief Termination status returned by @c fit().
 */
enum class Status {
  /** @brief A local Levenberg--Marquardt stopping criterion was met. */
  converged,

  /** @brief The residual evaluation budget was exhausted. */
  evaluation_limit,

  /** @brief A stopping tolerance became too small to make progress. */
  stalled,

  /** @brief The optimizer or a numerical calculation failed. */
  numerical_failure
};

/**
 * @brief Final nonlinear parameters, evaluation, and termination status.
 */
struct FitResult {
  /**
   * @brief Final nonlinear parameter vector.
   */
  Vector parameters;

  /**
   * @brief Evaluation corresponding to exactly @c parameters.
   */
  Evaluation evaluation;

  /**
   * @brief Termination status of the fit.
   */
  Status status = Status::numerical_failure;

  /**
   * @brief Number of Levenberg--Marquardt iterations.
   *
   * Starts at 1 for the initial point.
   */
  int iterations = 0;

  /**
   * @brief Number of residual callback evaluations, excluding the final
   * evaluation.
   */
  int function_evaluations = 0;

  /**
   * @brief Returns whether the fit ended with @ref Status::converged.
   */
  bool converged() const { return status == Status::converged; }
};

/**
 * @brief Evaluate the variable-projection problem at one parameter vector.
 *
 * The linear coefficients are solved by a numerical-rank compact SVD, and
 * the residual Jacobian is formed using @p jacobian_mode. No nonlinear
 * optimization is performed.
 *
 * @param problem Problem definition and model callbacks.
 * @param parameters Nonlinear parameter vector to evaluate.
 * @param linear_options Numerical-rank options for the linear solve.
 * @param jacobian_mode Formula used for the residual Jacobian.
 * @return Coefficients, weighted residuals, selected Jacobian, retained
 *         numerical rank, and SVD diagnostics at @p parameters.
 * @throws std::invalid_argument If the inputs, callback results' shapes, or
 *         linear options violate the API contract.
 * @throws std::domain_error If a model callback or numerical calculation
 *         produces a nonfinite result.
 * @note Exceptions thrown by user callbacks propagate to the caller.
 * @see CPP_DESIGN.md for the dimension, rank-cutoff, and Jacobian contracts.
 */
Evaluation evaluate(const Problem& problem, const Vector& parameters,
                    const LinearOptions& linear_options = {},
                    JacobianMode jacobian_mode = JacobianMode::kaufman);

/**
 * @brief Evaluate the problem with default linear options.
 *
 * @param problem Problem definition and model callbacks.
 * @param parameters Nonlinear parameter vector to evaluate.
 * @param jacobian_mode Formula used for the residual Jacobian.
 * @return Evaluation results at @p parameters.
 * @throws std::invalid_argument If the inputs or callback results' shapes
 *         violate the API contract.
 * @throws std::domain_error If a model callback or numerical calculation
 *         produces a nonfinite result.
 * @note Exceptions thrown by user callbacks propagate to the caller.
 */
Evaluation evaluate(const Problem& problem, const Vector& parameters,
                    JacobianMode jacobian_mode);

/**
 * @brief Fit the nonlinear parameters with Levenberg--Marquardt.
 *
 * At every trial parameter vector, the linear coefficients are eliminated by
 * the compact-SVD solve. The returned @ref FitResult::evaluation describes
 * the same final parameter vector as @ref FitResult::parameters, including
 * when the residual evaluation budget is exhausted.
 *
 * @param problem Problem definition and model callbacks.
 * @param initial_parameters Initial nonlinear parameter vector.
 * @param options Optimizer, Jacobian, and linear-solve options.
 * @return Final parameters, matching evaluation, termination status, and
 *         evaluation counters.
 * @throws std::invalid_argument If the inputs, options, callback results'
 *         shapes, or LM dimensions violate the API contract.
 * @throws std::domain_error If a model callback or numerical calculation
 *         produces a nonfinite result.
 * @note Exceptions thrown by user callbacks propagate to the caller.
 */
FitResult fit(const Problem& problem, Vector initial_parameters,
              const Options& options = {});

// ---- Header-only implementation ------------------------------------------------
namespace detail {

inline void validate(const Problem& p, const Vector& x) {
  if (p.observations.rows() <= 0 || p.observations.cols() <= 0)
    throw std::invalid_argument("observations must have positive dimensions");
  if (!p.basis || !p.derivative)
    throw std::invalid_argument("basis and derivative callbacks are required");
  if (x.size() <= 0 || !x.allFinite())
    throw std::invalid_argument("parameters must be nonempty and finite");
  if (!p.observations.allFinite())
    throw std::invalid_argument("observations must be finite");
  if (p.weights.size() && p.weights.size() != p.observations.rows())
    throw std::invalid_argument("weights must have one entry per row");
  if (p.weights.size()) {
    bool positive = false;
    for (Index i = 0; i < p.weights.size(); ++i) {
      if (!std::isfinite(p.weights(i)) || p.weights(i) < 0)
        throw std::invalid_argument("weights must be finite and nonnegative");
      positive |= p.weights(i) > 0;
    }
    if (!positive) throw std::invalid_argument("at least one weight must be positive");
  }
}

inline void validate_mode(JacobianMode mode) {
  if (mode != JacobianMode::kaufman && mode != JacobianMode::retained_subspace)
    throw std::invalid_argument("invalid Jacobian mode");
}

inline void validate_linear(const LinearOptions& options) {
  if (!std::isfinite(options.rcond)) throw std::invalid_argument("rcond must be finite");
}

inline Matrix weight_rows(Matrix a, const Vector& weights) {
  if (weights.size())
    for (Index j = 0; j < a.cols(); ++j)
      for (Index i = 0; i < a.rows(); ++i) a(i,j) *= weights(i);
  if (!a.allFinite()) throw std::domain_error("weighted matrix is nonfinite");
  return a;
}

struct SVD {
  Matrix u, v;
  Vector sigma;
};

// One-sided Jacobi: orthogonalize the columns directly, not via A^T A.
// Global scaling prevents column-norm overflow for finite inputs near DBL_MAX.
inline SVD thin_svd(const Matrix& a) {
  const Index m = a.rows(), n = a.cols();
  SVD result{Matrix(m,n), Matrix::Identity(n,n), Vector(n)};
  double scale = 0.0;
  for (Index i = 0; i < a.size(); ++i) scale = (std::max)(scale, std::abs(a.data()[i]));
  if (scale == 0) return result;
  Matrix b = a / scale;
  const double epsilon = 4.0 * std::numeric_limits<double>::epsilon();
  bool changed = false;
  const Index max_sweeps = (std::max)(Index(32), 8 * n);
  for (Index sweep = 0; sweep < max_sweeps; ++sweep) {
    changed = false;
    for (Index p = 0; p < n; ++p) for (Index q = p + 1; q < n; ++q) {
      // Normalize the two columns independently before forming their dot products.
      double p_scale = 0, q_scale = 0;
      for (Index i = 0; i < m; ++i) {
        p_scale = (std::max)(p_scale, std::abs(b(i,p)));
        q_scale = (std::max)(q_scale, std::abs(b(i,q)));
      }
      if (p_scale == 0 || q_scale == 0) continue;
      double aa = 0, bb = 0, ab = 0;
      for (Index i = 0; i < m; ++i) {
        const double x = b(i,p) / p_scale, y = b(i,q) / q_scale;
        aa += x * x;
        bb += y * y;
        ab += x * y;
      }
      const double p_unit_norm = std::sqrt(aa), q_unit_norm = std::sqrt(bb);
      const double correlation = (ab / p_unit_norm) / q_unit_norm;
      if (std::abs(correlation) <= epsilon) continue;
      const double p_norm = p_scale * p_unit_norm, q_norm = q_scale * q_unit_norm;
      const double ratio = (std::min)(p_norm, q_norm) / (std::max)(p_norm, q_norm);
      const double difference = std::copysign(1.0 - ratio * ratio, q_norm - p_norm);
      const double twice_cross = 2.0 * correlation * ratio;
      if (twice_cross == 0.0) continue;
      // Equivalent to the Jacobi tangent with (bb-aa)/(2*ab), without overflow.
      const double t = twice_cross /
                       (difference + std::copysign(std::hypot(difference, twice_cross),
                                                   difference));
      changed = true;
      const double c = 1.0 / std::hypot(1.0, t), s = t * c;
      for (Index i = 0; i < m; ++i) {
        const double x = b(i,p), y = b(i,q);
        b(i,p) = c*x - s*y; b(i,q) = s*x + c*y;
      }
      for (Index i = 0; i < n; ++i) {
        const double x = result.v(i,p), y = result.v(i,q);
        result.v(i,p) = c*x - s*y; result.v(i,q) = s*x + c*y;
      }
    }
    if (!changed) break;
  }
  if (changed) throw std::domain_error("Jacobi SVD did not converge");

  Vector norms(n);
  for (Index j = 0; j < n; ++j)
    for (Index i = 0; i < m; ++i) norms(j) = std::hypot(norms(j), b(i,j));
  // Selection sort avoids an index allocation and keeps aligned U/V columns.
  for (Index j = 0; j < n; ++j) {
    Index best = j;
    for (Index k = j + 1; k < n; ++k) if (norms(k) > norms(best)) best = k;
    if (best != j) {
      std::swap(norms(j), norms(best));
      for (Index i = 0; i < m; ++i) std::swap(b(i,j), b(i,best));
      for (Index i = 0; i < n; ++i) std::swap(result.v(i,j), result.v(i,best));
    }
    result.sigma(j) = norms(j) * scale;
    if (!std::isfinite(result.sigma(j)))
      throw std::domain_error("singular value overflow");
    if (norms(j) > 0)
      for (Index i = 0; i < m; ++i) result.u(i,j) = b(i,j) / norms(j);
  }
  return result;
}

struct Prepared {
  Index n = 0, rank = 0;
  Matrix coefficients, residuals, u, v;
  Vector sigma;
  Diagnostics diagnostics;
};

inline Prepared prepare(const Problem& p, const Vector& x,
                        const LinearOptions& options, Index expected_n = 0,
                        const Matrix* preweighted_y = nullptr) {
  validate(p, x);
  validate_linear(options);
  const Index m = p.observations.rows(), s = p.observations.cols();
  const Matrix basis = p.basis(x);
  if (!basis.allFinite()) throw std::domain_error("basis is nonfinite");
  if (basis.rows() != m || basis.cols() <= 0 || basis.cols() > m)
    throw std::invalid_argument("basis must have shape m x n with 0 < n <= m");
  if (expected_n && basis.cols() != expected_n)
    throw std::invalid_argument("basis column count changed during fit");
  const Matrix a = weight_rows(basis, p.weights);
  const Matrix y = preweighted_y ? *preweighted_y : weight_rows(p.observations, p.weights);
  const SVD svd = thin_svd(a);
  const Index n = basis.cols();
  const double max_sigma = svd.sigma(0);
  const double rcond = options.rcond >= 0 ? options.rcond
                       : static_cast<double>((std::max)(m,n)) * std::numeric_limits<double>::epsilon();
  Prepared result;
  result.n = n;
  result.diagnostics.singular_values = svd.sigma;
  result.diagnostics.sigma_max = max_sigma;
  // Compare normalized singular values to avoid multiplying DBL_MAX by rcond.
  for (Index i = 0; i < n; ++i)
    if (max_sigma > 0 && svd.sigma(i) / max_sigma > rcond) ++result.rank;
  result.u = svd.u.leftCols(result.rank);
  result.v = svd.v.leftCols(result.rank);
  result.sigma = Vector(result.rank);
  for (Index k = 0; k < result.rank; ++k) result.sigma(k) = svd.sigma(k);
  if (result.rank) {
    result.diagnostics.sigma_min_retained = result.sigma(result.rank - 1);
    result.diagnostics.condition_estimate = max_sigma / result.diagnostics.sigma_min_retained;
    if (!std::isfinite(result.diagnostics.condition_estimate))
      throw std::domain_error("condition estimate overflow");
  }
  result.coefficients = Matrix::Zero(n, s);
  for (Index j = 0; j < s; ++j) for (Index k = 0; k < result.rank; ++k) {
    double projection = 0;
    for (Index i = 0; i < m; ++i) projection += result.u(i,k) * y(i,j);
    const double factor = projection / result.sigma(k);
    for (Index i = 0; i < n; ++i) result.coefficients(i,j) += result.v(i,k) * factor;
  }
  if (!result.coefficients.allFinite())
    throw std::domain_error("linear coefficients are nonfinite");
  result.residuals = y - a * result.coefficients;
  if (!result.residuals.allFinite() || !std::isfinite(result.residuals.squaredNorm()))
    throw std::domain_error("residuals are nonfinite");
  return result;
}

inline Matrix jacobian(const Problem& p, const Vector& x,
                       const Prepared& prep, JacobianMode mode) {
  validate_mode(mode);
  const Index m = p.observations.rows(), s = p.observations.cols(), q = x.size();
  if (m > (std::numeric_limits<Index>::max)() / s)
    throw std::invalid_argument("stacked dimensions overflow");
  Matrix j(m*s,q);
  for (Index k = 0; k < q; ++k) {
    const Matrix derivative = p.derivative(x,k);
    if (!derivative.allFinite()) throw std::domain_error("derivative is nonfinite");
    if (derivative.rows() != m || derivative.cols() != prep.n)
      throw std::invalid_argument("derivative has invalid shape");
    const Matrix d = weight_rows(derivative, p.weights);
    const Matrix dc = d * prep.coefficients;
    Matrix column = prep.u * (prep.u.transpose() * dc) - dc;
    if (mode == JacobianMode::retained_subspace && prep.rank) {
      Matrix correction = prep.v.transpose() * (d.transpose() * prep.residuals);
      for (Index i = 0; i < prep.rank; ++i)
        for (Index l = 0; l < s; ++l) correction(i,l) /= prep.sigma(i);
      column -= prep.u * correction;
    }
    if (!column.allFinite()) throw std::domain_error("Jacobian is nonfinite");
    for (Index l = 0; l < s; ++l)
      for (Index i = 0; i < m; ++i) j(l*m+i,k) = column(i,l);
  }
  return j;
}

inline Evaluation evaluation(const Problem& p, const Vector& x,
                             const Prepared& prep, JacobianMode mode) {
  Evaluation r;
  r.rank = prep.rank;
  r.coefficients = prep.coefficients;
  r.residuals = prep.residuals;
  r.diagnostics = prep.diagnostics;
  r.jacobian = jacobian(p, x, prep, mode);
  return r;
}

}  // namespace detail

inline Evaluation evaluate(const Problem& p, const Vector& x,
                           const LinearOptions& opts, JacobianMode mode) {
  detail::validate_mode(mode);
  return detail::evaluation(p, x, detail::prepare(p, x, opts), mode);
}
inline Evaluation evaluate(const Problem& p, const Vector& x, JacobianMode mode) {
  return evaluate(p, x, LinearOptions{}, mode);
}

namespace detail {

// Solve the LM step as an augmented least-squares problem, avoiding J^T J
// and its squared condition number. Damping is scaled by Jacobian columns.
inline Vector damped_step(const Matrix& jac, const Matrix& residuals,
                          const Vector& diagonal, double lambda) {
  const Index m = jac.rows(), q = jac.cols();
  if (m > (std::numeric_limits<Index>::max)() - q)
    throw std::invalid_argument("augmented LM dimensions overflow");
  Matrix augmented(m + q, q);
  Vector rhs(m + q);
  for (Index i = 0; i < m; ++i) rhs(i) = -residuals.data()[i];
  for (Index k = 0; k < q; ++k) {
    for (Index i = 0; i < m; ++i) augmented(i,k) = jac(i,k);
    augmented(m+k,k) = std::sqrt(lambda) * std::sqrt(diagonal(k));
  }
  if (!augmented.allFinite()) throw std::domain_error("LM damping overflow");
  const SVD svd = thin_svd(augmented);
  Vector step(q);
  const double max_sigma = svd.sigma(0);
  for (Index k = 0; k < q; ++k) {
    if (max_sigma == 0 || svd.sigma(k) / max_sigma <=
        static_cast<double>(m+q) * std::numeric_limits<double>::epsilon()) continue;
    double projection = 0;
    for (Index i = 0; i < m+q; ++i) projection += svd.u(i,k) * rhs(i);
    const double scale = projection / svd.sigma(k);
    for (Index i = 0; i < q; ++i) step(i) += svd.v(i,k) * scale;
  }
  if (!step.allFinite()) throw std::domain_error("LM step is nonfinite");
  return step;
}

inline double predicted_reduction(const Matrix& jac, const Vector& gradient,
                                  const Vector& step) {
  return -gradient.dot(step) - 0.5 * (jac * step).squaredNorm();
}

}  // namespace detail

inline FitResult fit(const Problem& p, Vector initial, const Options& options) {
  detail::validate(p, initial);
  detail::validate_mode(options.jacobian_mode);
  detail::validate_linear(options.linear_options);
  if (options.max_evaluations <= 0 || !std::isfinite(options.ftol) ||
      !std::isfinite(options.xtol) || !std::isfinite(options.gtol) ||
      options.ftol < 0 || options.xtol < 0 || options.gtol < 0)
    throw std::invalid_argument("invalid fitting options");
  const Index m = p.observations.rows(), s = p.observations.cols(), q = initial.size();
  if (m > (std::numeric_limits<Index>::max)() / s || m*s < q)
    throw std::invalid_argument("fit requires m*s >= parameter count");
  const Matrix weighted_y = detail::weight_rows(p.observations, p.weights);
  Vector x = std::move(initial);
  detail::Prepared prepared = detail::prepare(p, x, options.linear_options, 0, &weighted_y);
  const Index basis_width = prepared.n;

  FitResult result;
  result.function_evaluations = 1;  // Initial residual callback.
  result.iterations = 1;
  result.status = Status::evaluation_limit;
  double lambda = 1e-3, nu = 2.0;
  // A separate iteration guard also covers repeated trial-step rejections.
  for (long long attempts = 0;
       attempts < (std::max)(100LL, 20LL * options.max_evaluations);
       ++attempts) {
    Matrix jac = detail::jacobian(p, x, prepared, options.jacobian_mode);
    double norm_r = prepared.residuals.norm();
    if (norm_r == 0) {
      result.status = Status::converged;
      break;
    }
    // A common scale preserves the LM model and keeps its quadratic terms
    // representable when the entire objective is scaled up or down.
    double scale = norm_r;
    for (Index i = 0; i < jac.size(); ++i)
      scale = (std::max)(scale, std::abs(jac.data()[i]));
    jac /= scale;
    const Matrix residuals = prepared.residuals / scale;
    norm_r /= scale;
    Vector gradient(q), diagonal(q);
    double max_diag = 0, max_cosine = 0;
    for (Index k = 0; k < q; ++k) {
      for (Index i = 0; i < m*s; ++i) {
        gradient(k) += jac(i,k) * residuals.data()[i];
        diagonal(k) += jac(i,k) * jac(i,k);
      }
      max_diag = (std::max)(max_diag, diagonal(k));
      if (norm_r > 0 && diagonal(k) > 0)
        max_cosine = (std::max)(max_cosine,
                         std::abs(gradient(k)) / (std::sqrt(diagonal(k)) * norm_r));
    }
    if (!gradient.allFinite() || !diagonal.allFinite())
      throw std::domain_error("LM normal equations are nonfinite");
    if (max_cosine <= options.gtol) {
      result.status = Status::converged;
      break;
    }
    if (result.function_evaluations >= options.max_evaluations) break;
    const double floor = max_diag * 1e-12;
    for (Index k = 0; k < q; ++k) diagonal(k) = (std::max)(diagonal(k), floor);
    Vector step = detail::damped_step(jac, residuals, diagonal, lambda);
    // A relative trust region prevents an otherwise valid decrease from
    // taking an enormous step into a different, ill-conditioned basin.
    double relative_step = 0;
    for (Index k = 0; k < q; ++k)
      relative_step = std::hypot(relative_step, step(k) / (1.0 + std::abs(x(k))));
    if (relative_step > 0.5) step *= 0.5 / relative_step;
    if (step.norm() <= options.xtol * (options.xtol + x.norm())) {
      result.status = Status::converged;
      break;
    }
    Vector trial = x + step;
    ++result.function_evaluations;
    detail::Prepared next = detail::prepare(p, trial, options.linear_options,
                                            basis_width, &weighted_y);
    const double next_norm = next.residuals.norm() / scale;
    const double predicted = detail::predicted_reduction(jac, gradient, step);
    if (predicted > 0 && next_norm < norm_r) {
      const double relative_decrease =
          ((norm_r - next_norm) / norm_r) * (1.0 + next_norm / norm_r);
      const double reduction = 0.5 * (norm_r - next_norm) * (norm_r + next_norm);
      const double ratio = reduction / predicted;
      x = std::move(trial);
      prepared = std::move(next);
      ++result.iterations;
      lambda *= (std::max)(1.0/3.0, 1.0 - std::pow(2.0*ratio-1.0, 3.0));
      lambda = (std::max)(lambda, 1e-30);
      nu = 2.0;
      if (relative_decrease <= options.ftol) {
        result.status = Status::converged;
        break;
      }
    } else {
      lambda *= nu;
      nu *= 2.0;
      if (!std::isfinite(lambda) || lambda > 1e30) {
        result.status = Status::stalled;
        break;
      }
    }
  }
  result.parameters = x;
  result.evaluation = detail::evaluation(p, x, prepared, options.jacobian_mode);
  return result;
}

}  // namespace varpro

#endif  // VARPRO_VARPRO_HPP_
