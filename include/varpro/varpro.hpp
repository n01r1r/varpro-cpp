#pragma once

#include <Eigen/Core>
#include <functional>

namespace varpro {

using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
using Vector = Eigen::VectorXd;

// Each observation column is a dataset; callbacks share parameters and coordinates.
struct Problem {
    Matrix observations;
    std::function<Matrix(const Vector&)> basis;
    std::function<Matrix(const Vector&, Eigen::Index)> derivative;
    Vector weights; // Empty = identity; otherwise shared residual multipliers (1/sigma).
};

enum class JacobianMode { kaufman, exact };

struct LinearOptions {
    double rcond = -1.0; // < 0 = automatic max(m, n) * epsilon cutoff.
};

struct Evaluation {
    Matrix coefficients;
    Matrix residuals; // Weighted residuals, one column per dataset.
    Matrix jacobian;  // Selected residual Jacobian; columns stack datasets vertically.
    Eigen::Index rank = 0;
    double squared_error() const { return residuals.squaredNorm(); }
};

struct Options {
    JacobianMode jacobian_mode = JacobianMode::kaufman;
    LinearOptions linear_options;
    int max_evaluations = 1000;
    double ftol = 1e-12;
    double xtol = 1e-12;
    double gtol = 1e-12;
};

enum class Status { converged, evaluation_limit, stalled, numerical_failure };

struct FitResult {
    Vector parameters;
    Evaluation evaluation;
    Status status = Status::numerical_failure;
    int iterations = 0; // Eigen LM counter; starts at 1 for the initial point.
    int function_evaluations = 0; // Eigen LM residual calls, excludes final evaluation.
    bool converged() const { return status == Status::converged; }
};

// Invalid contracts throw invalid_argument; nonfinite evaluations throw domain_error.
// See CPP_DESIGN.md for dimensions, rank cutoff, and the Jacobian formulas.
Evaluation evaluate(const Problem& problem, const Vector& parameters,
                    const LinearOptions& linear_options = {},
                    JacobianMode jacobian_mode = JacobianMode::kaufman);
Evaluation evaluate(const Problem& problem, const Vector& parameters,
                    JacobianMode jacobian_mode);
FitResult fit(const Problem& problem, Vector initial_parameters,
              const Options& options = {});

} // namespace varpro
