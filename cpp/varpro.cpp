#include <varpro/varpro.hpp>

#include <Eigen/SVD>
#include <unsupported/Eigen/LevenbergMarquardt>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>

namespace varpro {
namespace {

using Index = Eigen::Index;

bool finite(const Matrix& matrix) {
    return matrix.array().isFinite().all();
}

bool finite(const Vector& vector) {
    return vector.array().isFinite().all();
}

void validate_inputs(const Problem& problem, const Vector& parameters) {
    if (problem.observations.rows() <= 0 || problem.observations.cols() <= 0)
        throw std::invalid_argument("observations must have positive dimensions");
    if (!problem.basis || !problem.derivative)
        throw std::invalid_argument("basis and derivative callbacks are required");
    if (parameters.size() <= 0 || !finite(parameters))
        throw std::invalid_argument("parameters must be nonempty and finite");
    if (!finite(problem.observations))
        throw std::invalid_argument("observations must be finite");

    const Index m = problem.observations.rows();
    if (problem.weights.size() != 0 && problem.weights.size() != m)
        throw std::invalid_argument("weights must be empty or have one entry per observation");
    if (problem.weights.size() != 0) {
        if (!finite(problem.weights))
            throw std::invalid_argument("weights must be finite");
        bool any_positive = false;
        for (Index i = 0; i < m; ++i) {
            if (problem.weights(i) < 0.0)
                throw std::invalid_argument("weights must be nonnegative");
            any_positive = any_positive || problem.weights(i) > 0.0;
        }
        if (!any_positive)
            throw std::invalid_argument("at least one weight must be positive");
    }
}

Index stacked_size(Index rows, Index columns) {
    const Index maximum = (std::numeric_limits<Index>::max)();
    if (rows <= 0 || columns <= 0 || rows > maximum / columns)
        throw std::invalid_argument("stacked residual dimensions overflow Eigen::Index");
    return rows * columns;
}

Matrix apply_weights(const Matrix& matrix, const Vector& weights) {
    Matrix weighted = matrix;
    if (weights.size() == 0)
        return weighted;
    for (Index i = 0; i < weighted.rows(); ++i)
        weighted.row(i) *= weights(i);
    return weighted;
}

struct Prepared {
    Index n = 0;
    Index rank = 0;
    Matrix coefficients;
    Matrix residuals;
    Matrix retained_u;
    Matrix jacobian;
    bool has_jacobian = false;
};

Prepared prepare(const Problem& problem, const Vector& parameters, Index expected_n = 0) {
    validate_inputs(problem, parameters);

    const Index m = problem.observations.rows();
    const Index s = problem.observations.cols();
    const Matrix basis = problem.basis(parameters);
    if (!finite(basis))
        throw std::domain_error("basis callback returned nonfinite values");
    if (basis.rows() != m || basis.cols() <= 0 || basis.cols() > m)
        throw std::invalid_argument("basis must have shape m x n with 0 < n <= m");
    if (expected_n != 0 && basis.cols() != expected_n)
        throw std::invalid_argument("basis column count changed during fit");

    const Index n = basis.cols();
    const Matrix weighted_basis = apply_weights(basis, problem.weights);
    const Matrix weighted_observations = apply_weights(problem.observations, problem.weights);
    if (!finite(weighted_basis) || !finite(weighted_observations))
        throw std::domain_error("weighted model values are nonfinite");

    Eigen::JacobiSVD<Matrix> svd(
        weighted_basis, Eigen::ComputeThinU | Eigen::ComputeThinV);
    const Vector singular_values = svd.singularValues();
    if (!finite(singular_values))
        throw std::domain_error("SVD returned nonfinite singular values");

    const double sigma_max = singular_values.maxCoeff();
    const double cutoff = sigma_max *
        (static_cast<double>((std::max)(m, n)) *
         std::numeric_limits<double>::epsilon());
    Index rank = 0;
    for (Index i = 0; i < singular_values.size(); ++i)
        if (singular_values(i) > cutoff)
            ++rank;

    Prepared result;
    result.n = n;
    result.rank = rank;
    result.coefficients = Matrix::Zero(n, s);
    result.retained_u = Matrix(m, rank);
    if (rank != 0) {
        result.retained_u = svd.matrixU().leftCols(rank);
        const Matrix retained_v = svd.matrixV().leftCols(rank);
        const Vector retained_sigma = singular_values.head(rank);
        if (!finite(result.retained_u) || !finite(retained_v))
            throw std::domain_error("SVD returned nonfinite singular vectors");

        Matrix projected_rhs = result.retained_u.transpose() * weighted_observations;
        if (!finite(projected_rhs))
            throw std::domain_error("linear solve produced nonfinite values");
        for (Index i = 0; i < rank; ++i)
            projected_rhs.row(i) /= retained_sigma(i);
        if (!finite(projected_rhs))
            throw std::domain_error("linear solve produced nonfinite values");
        result.coefficients.noalias() = retained_v * projected_rhs;
    }
    if (!finite(result.coefficients))
        throw std::domain_error("linear solve produced nonfinite coefficients");

    result.residuals = weighted_observations - weighted_basis * result.coefficients;
    if (!finite(result.residuals))
        throw std::domain_error("residuals are nonfinite");
    const double squared_error = result.residuals.squaredNorm();
    if (!std::isfinite(squared_error))
        throw std::domain_error("squared error overflowed");
    return result;
}

Matrix form_jacobian(const Problem& problem, const Vector& parameters, Prepared& prepared) {
    const Index m = problem.observations.rows();
    const Index s = problem.observations.cols();
    const Index q = parameters.size();
    const Index stacked = stacked_size(m, s);
    Matrix jacobian(stacked, q);

    for (Index k = 0; k < q; ++k) {
        const Matrix derivative = problem.derivative(parameters, k);
        if (!finite(derivative))
            throw std::domain_error("derivative callback returned nonfinite values");
        if (derivative.rows() != m || derivative.cols() != prepared.n)
            throw std::invalid_argument("derivative must have the same m x n shape as basis");
        const Matrix weighted_derivative = apply_weights(derivative, problem.weights);
        if (!finite(weighted_derivative))
            throw std::domain_error("weighted derivative is nonfinite");

        Matrix column_matrix;
        if (s <= q) {
            const Matrix derivative_coefficients = weighted_derivative * prepared.coefficients;
            if (!finite(derivative_coefficients))
                throw std::domain_error("Jacobian intermediate is nonfinite");
            column_matrix = prepared.retained_u *
                (prepared.retained_u.transpose() * derivative_coefficients) -
                derivative_coefficients;
        } else {
            const Matrix projected_derivative = prepared.retained_u *
                (prepared.retained_u.transpose() * weighted_derivative) - weighted_derivative;
            if (!finite(projected_derivative))
                throw std::domain_error("Jacobian intermediate is nonfinite");
            column_matrix = projected_derivative * prepared.coefficients;
        }
        if (!finite(column_matrix))
            throw std::domain_error("Jacobian is nonfinite");
        for (Index j = 0; j < s; ++j)
            jacobian.col(k).segment(j * m, m) = column_matrix.col(j);
    }
    return jacobian;
}

bool same_vector(const Vector& left, const Vector& right) {
    return left.size() == right.size() &&
        (left.array() == right.array()).all();
}

class FitFunctor final : public Eigen::DenseFunctor<double> {
public:
    FitFunctor(const Problem& problem, const Options& options, int inputs, int values)
        : Eigen::DenseFunctor<double>(inputs, values), problem_(problem), options_(options),
          parameter_count_(inputs) {}

    int operator()(const InputType& parameters, ValueType& residuals) {
        if (function_evaluations_ >= options_.max_evaluations) {
            budget_exhausted_ = true;
            return -1;
        }
        ++function_evaluations_;
        Prepared& current = prepared_for(parameters);
        Eigen::Map<Matrix> output(residuals.data(), problem_.observations.rows(),
                                  problem_.observations.cols());
        output = current.residuals;
        return 0;
    }

    int df(const InputType& parameters, JacobianType& jacobian) {
        Prepared& current = prepared_for(parameters);
        if (!current.has_jacobian) {
            current.jacobian = form_jacobian(problem_, parameters, current);
            current.has_jacobian = true;
        }
        jacobian = current.jacobian;
        return 0;
    }

    int function_evaluations() const { return function_evaluations_; }
    bool budget_exhausted() const { return budget_exhausted_; }
    Index basis_columns() const { return expected_n_; }

private:
    Prepared& prepared_for(const Vector& parameters) {
        if (parameters.size() != parameter_count_)
            throw std::invalid_argument("parameter count changed during fit");
        if (cached_ && same_vector(cached_parameters_, parameters))
            return *cached_;

        Prepared fresh = prepare(problem_, parameters, expected_n_);
        if (expected_n_ == 0)
            expected_n_ = fresh.n;
        cached_parameters_ = parameters;
        cached_ = std::make_unique<Prepared>(std::move(fresh));
        return *cached_;
    }

    const Problem& problem_;
    const Options& options_;
    const Index parameter_count_;
    Index expected_n_ = 0;
    int function_evaluations_ = 0;
    bool budget_exhausted_ = false;
    Vector cached_parameters_;
    std::unique_ptr<Prepared> cached_;
};

Status map_status(Eigen::LevenbergMarquardtSpace::Status status, bool budget_exhausted) {
    using EigenStatus = Eigen::LevenbergMarquardtSpace::Status;
    if (budget_exhausted || status == EigenStatus::TooManyFunctionEvaluation)
        return Status::evaluation_limit;
    switch (status) {
    case EigenStatus::RelativeReductionTooSmall:
    case EigenStatus::RelativeErrorTooSmall:
    case EigenStatus::RelativeErrorAndReductionTooSmall:
    case EigenStatus::CosinusTooSmall:
        return Status::converged;
    case EigenStatus::FtolTooSmall:
    case EigenStatus::XtolTooSmall:
    case EigenStatus::GtolTooSmall:
        return Status::stalled;
    default:
        return Status::numerical_failure;
    }
}

} // namespace

Evaluation evaluate(const Problem& problem, const Vector& parameters) {
    Prepared prepared = prepare(problem, parameters);
    Matrix jacobian = form_jacobian(problem, parameters, prepared);
    Evaluation result;
    result.coefficients = std::move(prepared.coefficients);
    result.residuals = std::move(prepared.residuals);
    result.jacobian = std::move(jacobian);
    result.rank = prepared.rank;
    return result;
}

FitResult fit(const Problem& problem, Vector initial_parameters, const Options& options) {
    validate_inputs(problem, initial_parameters);
    if (options.max_evaluations <= 0 || !std::isfinite(options.ftol) ||
        !std::isfinite(options.xtol) || !std::isfinite(options.gtol) ||
        options.ftol < 0.0 || options.xtol < 0.0 || options.gtol < 0.0)
        throw std::invalid_argument("invalid fitting options");

    const Index q = initial_parameters.size();
    const Index stacked = stacked_size(problem.observations.rows(), problem.observations.cols());
    const Index int_max = (std::numeric_limits<int>::max)();
    if (q > int_max || stacked > int_max || stacked < q)
        throw std::invalid_argument("fit dimensions are incompatible with Eigen LM");

    FitFunctor functor(problem, options, static_cast<int>(q), static_cast<int>(stacked));
    Eigen::LevenbergMarquardt<FitFunctor> solver(functor);
    solver.setMaxfev(options.max_evaluations);
    solver.setFtol(options.ftol);
    solver.setXtol(options.xtol);
    solver.setGtol(options.gtol);
    Vector parameters = std::move(initial_parameters);
    const Eigen::LevenbergMarquardtSpace::Status lm_status = solver.minimize(parameters);

    FitResult result;
    result.parameters = parameters;
    result.evaluation = evaluate(problem, parameters);
    if (functor.basis_columns() != 0 &&
        result.evaluation.coefficients.rows() != functor.basis_columns())
        throw std::invalid_argument("basis column count changed during fit");
    result.status = map_status(lm_status, functor.budget_exhausted());
    result.iterations = static_cast<int>(solver.iterations());
    result.function_evaluations = functor.function_evaluations();
    return result;
}

} // namespace varpro
