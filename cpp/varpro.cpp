#include <varpro/varpro.hpp>

#include <Eigen/SVD>
#include <unsupported/Eigen/LevenbergMarquardt>

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>

namespace varpro {
namespace {

using Index = Eigen::Index;

// Callback failures are distinguished from invalid user input.  The public
// API uses invalid_argument for a contract violation detected before or while
// checking inputs, and domain_error for a nonfinite value produced by the
// numerical model or a linear-algebra operation.
bool finite(const Matrix& matrix) {
    return matrix.array().isFinite().all();
}

bool finite(const Vector& vector) {
    return vector.array().isFinite().all();
}

void validate_inputs(const Problem& problem, const Vector& parameters) {
    // Validate the dimensions that are needed by every subsequent callback
    // and matrix operation before calling user-supplied model functions.
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
        // A weight is a residual multiplier, not a squared-error multiplier:
        // the weighted problem uses W = diag(weights), so each row is scaled
        // once before the residual norm is squared.
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

void validate_linear_options(const LinearOptions& options) {
    if (!std::isfinite(options.rcond))
        throw std::invalid_argument("rcond must be finite");
}

void validate_jacobian_mode(JacobianMode mode) {
    if (mode != JacobianMode::kaufman && mode != JacobianMode::exact)
        throw std::invalid_argument("invalid Jacobian mode");
}

Index stacked_size(Index rows, Index columns) {
    // Eigen's LM interface consumes one flat residual vector.  The public API
    // keeps residuals as an m x s matrix, so this is the size of the
    // column-major vector [residuals.col(0); residuals.col(1); ...].
    // Check before multiplying because Eigen::Index is signed and the product
    // is later passed through Eigen's int-sized LM interface.
    const Index maximum = (std::numeric_limits<Index>::max)();
    if (rows <= 0 || columns <= 0 || rows > maximum / columns)
        throw std::invalid_argument("stacked residual dimensions overflow Eigen::Index");
    return rows * columns;
}

Matrix apply_weights(const Matrix& matrix, const Vector& weights) {
    // Left multiplication by diag(weights) is equivalent to multiplying each
    // row.  Avoid materializing an m x m diagonal matrix because the only
    // operation needed is this row scaling.
    Matrix weighted = matrix;
    if (weights.size() == 0)
        return weighted;
    for (Index i = 0; i < weighted.rows(); ++i)
        weighted.row(i) *= weights(i);
    return weighted;
}

Matrix make_weighted_observations(const Problem& problem) {
    Matrix weighted = apply_weights(problem.observations, problem.weights);
    if (!finite(weighted))
        throw std::domain_error("weighted observations are nonfinite");
    return weighted;
}

// State computed for one nonlinear parameter vector.  The retained compact
// SVD factors are kept alongside C and R so that the linear solve, range
// projector, and Jacobian all use exactly the same numerical rank decision.
struct Prepared {
    Index n = 0;
    Index rank = 0;
    Matrix coefficients;
    Matrix residuals;
    Matrix retained_u;
    Matrix retained_v;
    Vector retained_sigma;
    Matrix jacobian;
    bool has_jacobian = false;
};

Prepared prepare(const Problem& problem, const Vector& parameters,
                 const LinearOptions& linear_options, Index expected_n = 0,
                 const Matrix* cached_weighted_observations = nullptr) {
    // This is the variable-projection inner solve.  For a fixed alpha, form
    //     A(alpha) = W * Phi(alpha),  B = W * Y,
    // solve C(alpha) = A(alpha)^+ B, and return R = B - A(alpha) C(alpha).
    // B is supplied by fit when possible because it does not depend on alpha.
    validate_inputs(problem, parameters);
    validate_linear_options(linear_options);

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
    Matrix computed_weighted_observations;
    const Matrix* weighted_observations = cached_weighted_observations;
    if (weighted_observations == nullptr) {
        computed_weighted_observations = make_weighted_observations(problem);
        weighted_observations = &computed_weighted_observations;
    }
    if (weighted_observations->rows() != m || weighted_observations->cols() != s ||
        !finite(*weighted_observations))
        throw std::domain_error("weighted observations are nonfinite or have invalid shape");
    if (!finite(weighted_basis))
        throw std::domain_error("weighted model values are nonfinite");

    // Thin U and V contain all factors needed by the minimum-norm solve and
    // the Jacobian, while avoiding a full m x m projector or explicit
    // pseudoinverse.
    Eigen::JacobiSVD<Matrix> svd(
        weighted_basis, Eigen::ComputeThinU | Eigen::ComputeThinV);
    const Vector singular_values = svd.singularValues();
    if (!finite(singular_values))
        throw std::domain_error("SVD returned nonfinite singular values");

    const double sigma_max = singular_values.maxCoeff();
    const double relative_cutoff = linear_options.rcond < 0.0
        ? static_cast<double>((std::max)(m, n)) * std::numeric_limits<double>::epsilon()
        : linear_options.rcond;

    // Eigen orders singular values from largest to smallest.  Retaining the
    // leading values above the relative cutoff therefore gives a compact
    // numerical-rank factorization.  A zero sigma_max means that A is the
    // zero matrix; rank zero is valid and yields C = 0 and J = 0.
    Index rank = 0;
    for (Index i = 0; i < singular_values.size(); ++i)
        if (sigma_max > 0.0 && singular_values(i) / sigma_max > relative_cutoff)
            ++rank;

    Prepared result;
    result.n = n;
    result.rank = rank;
    result.coefficients = Matrix::Zero(n, s);
    result.retained_u = Matrix(m, rank);
    result.retained_v = Matrix(n, rank);
    result.retained_sigma = Vector(rank);
    if (rank != 0) {
        result.retained_u = svd.matrixU().leftCols(rank);
        result.retained_v = svd.matrixV().leftCols(rank);
        result.retained_sigma = singular_values.head(rank);
        if (!finite(result.retained_u) || !finite(result.retained_v) ||
            !finite(result.retained_sigma))
            throw std::domain_error("SVD returned nonfinite singular vectors");

        // With A ~= U_r Sigma_r V_r^T, the truncated minimum-norm solve is
        // C = V_r Sigma_r^-1 U_r^T B.  Dividing the projected RHS in place
        // avoids constructing an explicit pseudoinverse.
        Matrix projected_rhs = result.retained_u.transpose() * *weighted_observations;
        if (!finite(projected_rhs))
            throw std::domain_error("linear solve produced nonfinite values");
        for (Index i = 0; i < rank; ++i)
            projected_rhs.row(i) /= result.retained_sigma(i);
        if (!finite(projected_rhs))
            throw std::domain_error("linear solve produced nonfinite values");
        result.coefficients.noalias() = result.retained_v * projected_rhs;
    }
    if (!finite(result.coefficients))
        throw std::domain_error("linear solve produced nonfinite coefficients");

    // Use the same truncated coefficients for the residual that were used in
    // the solve.  This keeps the reported residual and Jacobian state
    // consistent at a rank-deficient or explicitly truncated basis.
    result.residuals = *weighted_observations - weighted_basis * result.coefficients;
    if (!finite(result.residuals))
        throw std::domain_error("residuals are nonfinite");
    const double squared_error = result.residuals.squaredNorm();
    if (!std::isfinite(squared_error))
        throw std::domain_error("squared error overflowed");
    return result;
}

Matrix form_jacobian(const Problem& problem, const Vector& parameters, Prepared& prepared,
                     JacobianMode jacobian_mode) {
    validate_jacobian_mode(jacobian_mode);
    const Index m = problem.observations.rows();
    const Index s = problem.observations.cols();
    const Index q = parameters.size();
    const Index stacked = stacked_size(m, s);
    Matrix jacobian(stacked, q);

    // For each nonlinear parameter alpha[k], D_k = W * dPhi/dalpha[k].
    // The Kaufman approximation for R = B - A*C is
    //     J_k = -(I - U_r U_r^T) D_k C.
    // The matrix below is kept as m x s and is copied into one stacked
    // Jacobian column after all datasets have been handled.
    // Column stacking matches Eigen's column-major residual layout.
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
        // Both branches compute -(I - U_r U_r^T) * D_k * C.  Choosing the
        // multiplication order based on s and n avoids needlessly forming a
        // large intermediate when there are many right-hand sides.
        // This reassociation preserves the result while reducing temporary size.
        if (s <= prepared.n) {
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
        if (jacobian_mode == JacobianMode::exact && prepared.rank != 0) {
            // The exact retained-subspace formula adds the response of the
            // eliminated linear coefficients:
            //     J_exact = J_Kaufman
            //               - U_r Sigma_r^-1 V_r^T D_k^T R.
            // The subtraction sign follows the residual convention B - A*C.
            Matrix correction = weighted_derivative.transpose() * prepared.residuals;
            if (!finite(correction))
                throw std::domain_error("exact Jacobian intermediate is nonfinite");
            correction = prepared.retained_v.transpose() * correction;
            if (!finite(correction))
                throw std::domain_error("exact Jacobian intermediate is nonfinite");
            for (Index i = 0; i < prepared.rank; ++i)
                correction.row(i) /= prepared.retained_sigma(i);
            if (!finite(correction))
                throw std::domain_error("exact Jacobian intermediate is nonfinite");
            column_matrix.noalias() -= prepared.retained_u * correction;
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

// Adapter between the matrix-oriented VarPro API and Eigen's LM callbacks.
// Eigen asks for residuals and derivatives separately, so this object caches
// the complete Prepared state for the most recently requested alpha.
class FitFunctor final : public Eigen::DenseFunctor<double> {
public:
    FitFunctor(const Problem& problem, const Options& options, int inputs, int values,
               Matrix weighted_observations)
        : Eigen::DenseFunctor<double>(inputs, values), problem_(problem), options_(options),
          parameter_count_(inputs), weighted_observations_(std::move(weighted_observations)) {}

    int operator()(const InputType& parameters, ValueType& residuals) {
        // Count only residual callbacks against the public evaluation budget.
        // Returning -1 makes Eigen stop, while the explicit guard gives the
        // library a strict one-call limit even if Eigen's internal counter
        // would otherwise request one more evaluation.
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
        // A Jacobian request at the same alpha reuses the SVD and residual
        // computed by operator().  If Eigen asks for df first, the same helper
        // computes the state once and returns its Jacobian.
        Prepared& current = prepared_with_jacobian(parameters);
        jacobian = current.jacobian;
        return 0;
    }

    int function_evaluations() const { return function_evaluations_; }
    bool budget_exhausted() const { return budget_exhausted_; }
    Index basis_columns() const { return expected_n_; }

    Evaluation evaluation_for(const Vector& parameters) {
        // The final evaluation is deliberately outside the LM callback count:
        // FitResult must describe the exact final parameter vector, including
        // a fit that stopped because its residual-evaluation budget expired.
        Prepared& current = prepared_with_jacobian(parameters);
        Evaluation result;
        result.coefficients = current.coefficients;
        result.residuals = current.residuals;
        result.jacobian = current.jacobian;
        result.rank = current.rank;
        return result;
    }

private:
    Prepared& prepared_with_jacobian(const Vector& parameters) {
        Prepared& current = prepared_for(parameters);
        if (!current.has_jacobian) {
            // Compute all derivative callbacks lazily.  Residual-only users do
            // not pay for Jacobians, and repeated LM requests at one point do
            // not repeat either the callbacks or the SVD.
            current.jacobian = form_jacobian(problem_, parameters, current,
                                             options_.jacobian_mode);
            current.has_jacobian = true;
        }
        return current;
    }

    Prepared& prepared_for(const Vector& parameters) {
        if (parameters.size() != parameter_count_)
            throw std::invalid_argument("parameter count changed during fit");
        // LM commonly requests the residual and Jacobian at the same point.
        // Exact vector equality is intentional here: a different point must
        // get a fresh basis/SVD, while a repeated point can reuse all state.
        if (cached_ && same_vector(cached_parameters_, parameters))
            return *cached_;

        Prepared fresh = prepare(problem_, parameters, options_.linear_options, expected_n_,
                                 &weighted_observations_);
        // The basis width is part of the callback contract and must remain
        // fixed throughout one fit because Eigen's residual/Jacobian shapes
        // were established from the first valid evaluation.
        if (expected_n_ == 0)
            expected_n_ = fresh.n;
        cached_parameters_ = parameters;
        cached_ = std::move(fresh);
        return *cached_;
    }

    const Problem& problem_;
    const Options& options_;
    const Index parameter_count_;
    Matrix weighted_observations_;
    Index expected_n_ = 0;
    int function_evaluations_ = 0;
    bool budget_exhausted_ = false;
    Vector cached_parameters_;
    std::optional<Prepared> cached_;
};

Status map_status(Eigen::LevenbergMarquardtSpace::Status status, bool budget_exhausted) {
    // Keep Eigen-specific termination details private.  In particular, an
    // explicit callback budget exhaustion takes precedence over an LM status
    // that might otherwise look like convergence.
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

Evaluation evaluate(const Problem& problem, const Vector& parameters,
                    const LinearOptions& linear_options, JacobianMode jacobian_mode) {
    // Standalone evaluation follows the same inner solve as fit, but without
    // an optimizer or cache.  It is useful when another optimizer owns alpha.
    validate_jacobian_mode(jacobian_mode);
    Prepared prepared = prepare(problem, parameters, linear_options);
    Matrix jacobian = form_jacobian(problem, parameters, prepared, jacobian_mode);
    Evaluation result;
    result.coefficients = std::move(prepared.coefficients);
    result.residuals = std::move(prepared.residuals);
    result.jacobian = std::move(jacobian);
    result.rank = prepared.rank;
    return result;
}

Evaluation evaluate(const Problem& problem, const Vector& parameters,
                    JacobianMode jacobian_mode) {
    return evaluate(problem, parameters, LinearOptions{}, jacobian_mode);
}

FitResult fit(const Problem& problem, Vector initial_parameters, const Options& options) {
    // LM optimizes only the nonlinear parameters.  Linear coefficients are
    // eliminated by prepare() at each accepted/trial parameter vector.
    validate_inputs(problem, initial_parameters);
    validate_jacobian_mode(options.jacobian_mode);
    validate_linear_options(options.linear_options);
    if (options.max_evaluations <= 0 || !std::isfinite(options.ftol) ||
        !std::isfinite(options.xtol) || !std::isfinite(options.gtol) ||
        options.ftol < 0.0 || options.xtol < 0.0 || options.gtol < 0.0)
        throw std::invalid_argument("invalid fitting options");

    const Index q = initial_parameters.size();
    const Index stacked = stacked_size(problem.observations.rows(), problem.observations.cols());
    const Index int_max = (std::numeric_limits<int>::max)();
    if (q > int_max || stacked > int_max || stacked < q)
        throw std::invalid_argument("fit dimensions are incompatible with Eigen LM");

    // B = W*Y is constant during the fit, so construct it once and pass it to
    // every prepared evaluation instead of reweighting the observations on
    // every LM callback.
    Matrix weighted_observations = make_weighted_observations(problem);
    FitFunctor functor(problem, options, static_cast<int>(q), static_cast<int>(stacked),
                       std::move(weighted_observations));
    Eigen::LevenbergMarquardt<FitFunctor> solver(functor);
    solver.setMaxfev(options.max_evaluations);
    solver.setFtol(options.ftol);
    solver.setXtol(options.xtol);
    solver.setGtol(options.gtol);
    Vector parameters = std::move(initial_parameters);
    const Eigen::LevenbergMarquardtSpace::Status lm_status = solver.minimize(parameters);

    FitResult result;
    // Ask for the final state after minimize() so parameters, coefficients,
    // residuals, rank, and Jacobian all refer to one identical point.
    result.parameters = parameters;
    result.evaluation = functor.evaluation_for(parameters);
    if (functor.basis_columns() != 0 &&
        result.evaluation.coefficients.rows() != functor.basis_columns())
        throw std::invalid_argument("basis column count changed during fit");
    result.status = map_status(lm_status, functor.budget_exhausted());
    result.iterations = static_cast<int>(solver.iterations());
    result.function_evaluations = functor.function_evaluations();
    return result;
}

} // namespace varpro
