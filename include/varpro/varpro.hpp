#pragma once

#include <Eigen/Core>
#include <functional>

/**
 * @file varpro.hpp
 * @brief Public API for variable projection of separable nonlinear
 *        least-squares problems.
 */

namespace varpro {

/**
 * @brief Column-major dynamic-size double-precision matrix.
 */
using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

/**
 * @brief Dynamic-size double-precision vector.
 */
using Vector = Eigen::VectorXd;

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
 * - @c observations has shape @f$m \times s@f$ with @f$m > 0@f$ and @f$s > 0@f$.
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
    std::function<Matrix(const Vector&, Eigen::Index)> derivative;

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
     * This is the exact VarPro formula for a locally constant retained rank.
     * If an explicit cutoff discards a nonzero singular direction, the
     * discarded direction is treated as null by the implementation.
     */
    exact
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
    Eigen::Index rank = 0;

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
     * This uses Eigen's iteration counter and starts at 1 for the initial
     * point.
     */
    int iterations = 0;

    /**
     * @brief Number of residual callback evaluations, excluding the final evaluation.
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
 * @return Coefficients, weighted residuals, selected Jacobian, and retained
 *         numerical rank at @p parameters.
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
 *         shapes, or Eigen LM dimensions violate the API contract.
 * @throws std::domain_error If a model callback or numerical calculation
 *         produces a nonfinite result.
 * @note Exceptions thrown by user callbacks propagate to the caller.
 */
FitResult fit(const Problem& problem, Vector initial_parameters,
              const Options& options = {});

} // namespace varpro
