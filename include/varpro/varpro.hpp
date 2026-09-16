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

struct Evaluation {
    Matrix coefficients;
    Matrix residuals; // Weighted residuals, one column per dataset.
    Matrix jacobian;  // Kaufman approximation; residual columns stacked vertically.
    Eigen::Index rank = 0;
    double squared_error() const { return residuals.squaredNorm(); }
};

struct Options {
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
// See CPP_DESIGN.md for dimensions, rank cutoff, and the approximate Jacobian.
Evaluation evaluate(const Problem& problem, const Vector& parameters);
FitResult fit(const Problem& problem, Vector initial_parameters,
              const Options& options = {});

} // namespace varpro
