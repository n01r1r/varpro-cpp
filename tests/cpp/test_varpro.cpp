#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <varpro/varpro.hpp>

namespace {

using varpro::Matrix;
using varpro::JacobianMode;
using varpro::LinearOptions;
using varpro::Problem;
using varpro::Vector;

void check(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void close(double actual, double expected, double tolerance, const std::string& message) {
    if (!std::isfinite(actual) || !std::isfinite(expected) || std::abs(actual - expected) > tolerance)
        throw std::runtime_error(message + " (actual=" + std::to_string(actual) + ")");
}

void close_matrix(const Matrix& actual, const Matrix& expected, double tolerance,
                  const std::string& message) {
    check(actual.rows() == expected.rows() && actual.cols() == expected.cols(), message + " shape");
    check(actual.allFinite() && expected.allFinite(), message + " finite");
    check((actual - expected).cwiseAbs().maxCoeff() <= tolerance, message);
}

Problem exponential_problem(const Vector& t, const Matrix& observations, bool offset = true) {
    Problem p;
    p.observations = observations;
    p.basis = [t, offset](const Vector& a) {
        Matrix phi(t.size(), offset ? 3 : 2);
        for (Eigen::Index i = 0; i < t.size(); ++i) {
            phi(i, 0) = std::exp(-t(i) / a(0));
            phi(i, 1) = std::exp(-t(i) / a(1));
            if (offset) phi(i, 2) = 1.0;
        }
        return phi;
    };
    p.derivative = [t, offset](const Vector& a, Eigen::Index k) {
        Matrix d = Matrix::Zero(t.size(), offset ? 3 : 2);
        if (k < 2) {
            for (Eigen::Index i = 0; i < t.size(); ++i)
                d(i, k) = std::exp(-t(i) / a(k)) * t(i) / (a(k) * a(k));
        }
        return d;
    };
    return p;
}

Matrix exp_data(const Vector& t, const Vector& taus, const Matrix& coefficients) {
    Matrix phi(t.size(), taus.size() + 1);
    for (Eigen::Index i = 0; i < t.size(); ++i) {
        for (Eigen::Index k = 0; k < taus.size(); ++k)
            phi(i, k) = std::exp(-t(i) / taus(k));
        phi(i, taus.size()) = 1.0;
    }
    return phi * coefficients;
}

void check_exp_fit(const varpro::FitResult& result, const Matrix& expected_coefficients,
                   const std::string& label) {
    check(result.converged(), label + " did not converge");
    const bool direct = std::abs(result.parameters(0) - 1.0) < 2e-6 &&
                        std::abs(result.parameters(1) - 3.0) < 2e-6;
    const bool swapped = std::abs(result.parameters(0) - 3.0) < 2e-6 &&
                         std::abs(result.parameters(1) - 1.0) < 2e-6;
    check(direct || swapped, label + " nonlinear parameters");
    Matrix coefficients = expected_coefficients;
    if (swapped) {
        coefficients.row(0) = expected_coefficients.row(1);
        coefficients.row(1) = expected_coefficients.row(0);
    }
    close_matrix(result.evaluation.coefficients, coefficients, 2e-6, label + " coefficients");
    close_matrix(result.evaluation.residuals, Matrix::Zero(result.evaluation.residuals.rows(),
                                                            result.evaluation.residuals.cols()),
                 2e-8, label + " residuals");
}

void test_single_and_mrhs_fit() {
    Vector t(11);
    for (Eigen::Index i = 0; i < t.size(); ++i) t(i) = static_cast<double>(i);
    Vector truth(2); truth << 1.0, 3.0;
    Matrix c(3, 3); c << 4.0, 2.0, -1.0, 2.5, 3.5, .75, 1.0, -0.5, 2.25;
    Matrix y = exp_data(t, truth, c);
    Vector initial(2); initial << 2.0, 6.5;
    Matrix single_y = y.leftCols(1);
    auto single = varpro::fit(exponential_problem(t, single_y), initial);
    check_exp_fit(single, c.leftCols(1), "single RHS fit");

    Problem p = exponential_problem(t, y);
    auto result = varpro::fit(p, initial);
    check_exp_fit(result, c, "MRHS fit");

    varpro::Options exact_options;
    exact_options.jacobian_mode = JacobianMode::exact;
    auto exact_result = varpro::fit(p, initial, exact_options);
    check_exp_fit(exact_result, c, "exact-Jacobian MRHS fit");
}

void test_weighting_and_orthogonality() {
    Vector t(8); for (Eigen::Index i = 0; i < t.size(); ++i) t(i) = static_cast<double>(i) * .7;
    Vector a(2); a << 1.4, 4.2;
    Matrix c(3, 1); c << 2.0, -1.0, .7;
    Matrix y = exp_data(t, a, c);
    y(2, 0) += .11; y(6, 0) -= .08;
    Vector w(8); w << 1., .5, 2., 1.5, .8, 1.2, .7, 1.8;
    Problem weighted = exponential_problem(t, y);
    weighted.weights = w;
    auto e = varpro::evaluate(weighted, a);
    Matrix phi = weighted.basis(a);
    Matrix aw = phi;
    Matrix rw = e.residuals;
    for (Eigen::Index i = 0; i < t.size(); ++i) { aw.row(i) *= w(i); }
    close_matrix(aw.transpose() * rw, Matrix::Zero(3, 1), 2e-12, "weighted residual orthogonality");

    Problem preweighted = exponential_problem(t, y, true);
    preweighted.observations = y;
    preweighted.basis = [base = weighted.basis, w](const Vector& x) {
        Matrix z = base(x); for (Eigen::Index i = 0; i < z.rows(); ++i) z.row(i) *= w(i); return z;
    };
    preweighted.derivative = [deriv = weighted.derivative, w](const Vector& x, Eigen::Index k) {
        Matrix z = deriv(x, k); for (Eigen::Index i = 0; i < z.rows(); ++i) z.row(i) *= w(i); return z;
    };
    for (Eigen::Index i = 0; i < y.rows(); ++i) preweighted.observations.row(i) *= w(i);
    auto ep = varpro::evaluate(preweighted, a);
    close_matrix(e.coefficients, ep.coefficients, 2e-12, "manual preweight coefficients");
    close_matrix(e.residuals, ep.residuals, 2e-12, "manual preweight residuals");
    close_matrix(e.jacobian, ep.jacobian, 2e-11, "manual preweight Jacobian");

    Vector initial(2); initial << 1.8, 5.5;
    auto weighted_fit = varpro::fit(weighted, initial);
    auto preweighted_fit = varpro::fit(preweighted, initial);
    check(weighted_fit.converged() && preweighted_fit.converged(), "weighted fit convergence");
    close_matrix(weighted_fit.parameters, preweighted_fit.parameters, 2e-8,
                 "manual preweight fitted parameters");
    close_matrix(weighted_fit.evaluation.coefficients, preweighted_fit.evaluation.coefficients,
                 2e-8, "manual preweight fitted coefficients");
    close_matrix(weighted_fit.evaluation.residuals, preweighted_fit.evaluation.residuals,
                 2e-8, "manual preweight fitted residuals");
}

void check_residual_jacobian(const Problem& problem, const Vector& parameters,
                             const LinearOptions& linear_options, JacobianMode mode,
                             double tolerance, const std::string& label) {
    const auto evaluation = varpro::evaluate(problem, parameters, linear_options, mode);
    const double h = 1e-6;
    Matrix fd(evaluation.jacobian.rows(), parameters.size());
    const Eigen::Index m = problem.observations.rows();
    const Eigen::Index s = problem.observations.cols();
    for (Eigen::Index k = 0; k < parameters.size(); ++k) {
        Vector plus = parameters, minus = parameters;
        plus(k) += h; minus(k) -= h;
        Matrix difference = varpro::evaluate(problem, plus, linear_options, mode).residuals -
                            varpro::evaluate(problem, minus, linear_options, mode).residuals;
        for (Eigen::Index j = 0; j < s; ++j)
            fd.col(k).segment(j * m, m) = difference.col(j) / (2.0 * h);
    }
    check((fd - evaluation.jacobian).cwiseAbs().maxCoeff() < tolerance, label);
}

void test_kaufman_jacobians() {
    Vector t(9); for (Eigen::Index i = 0; i < t.size(); ++i) t(i) = i * .6;
    Vector a(2); a << 1.2, 3.7;
    Matrix c(3, 1); c << 2.3, -1.1, .4;
    Problem exact = exponential_problem(t, exp_data(t, a, c));
    const LinearOptions linear_options;
    check_residual_jacobian(exact, a, linear_options, JacobianMode::kaufman, 3e-7,
                            "Kaufman zero-residual Jacobian finite difference");

    Matrix mrhs_coefficients(3, 3);
    mrhs_coefficients << 2.3, -.7, 1.1, -1.1, 2.0, .3, .4, .6, -.2;
    Problem exact_mrhs = exponential_problem(t, exp_data(t, a, mrhs_coefficients));
    check_residual_jacobian(exact_mrhs, a, linear_options, JacobianMode::kaufman, 3e-7,
                            "Kaufman zero-residual MRHS Jacobian finite difference");

    Problem off = exponential_problem(t, exp_data(t, a, c));
    off.observations(3, 0) += .13; off.observations(7, 0) -= .09;
    Vector x(2); x << 1.8, 5.5;
    auto eo = varpro::evaluate(off, x, linear_options, JacobianMode::kaufman);
    const double h = 1e-6;
    for (Eigen::Index k = 0; k < 2; ++k) {
        Vector plus = x, minus = x; plus(k) += h; minus(k) -= h;
        double fd_objective = (varpro::evaluate(off, plus, linear_options,
                                                 JacobianMode::kaufman).squared_error() -
                                varpro::evaluate(off, minus, linear_options,
                                                 JacobianMode::kaufman).squared_error()) / (2.0 * h);
        double jacobian_objective = 2.0 * eo.residuals.col(0).dot(eo.jacobian.col(k));
        close(jacobian_objective, fd_objective, 2e-6,
              "Kaufman nonzero-residual objective gradient");
    }
}

void test_exact_jacobians() {
    Vector t(9); for (Eigen::Index i = 0; i < t.size(); ++i) t(i) = i * .6;
    Vector a(2); a << 1.2, 3.7;
    Matrix c(3, 1); c << 2.3, -1.1, .4;
    const LinearOptions linear_options;

    Problem exact = exponential_problem(t, exp_data(t, a, c));
    check_residual_jacobian(exact, a, linear_options, JacobianMode::exact, 3e-7,
                            "exact zero-residual Jacobian finite difference");

    Matrix mrhs_coefficients(3, 4);
    mrhs_coefficients << 2.3, -.7, 1.1, .2,
                         -1.1, 2.0, .3, -.4,
                         .4, .6, -.2, 1.7;
    Problem exact_mrhs = exponential_problem(t, exp_data(t, a, mrhs_coefficients));
    check_residual_jacobian(exact_mrhs, a, linear_options, JacobianMode::exact, 3e-7,
                            "exact zero-residual MRHS Jacobian finite difference");

    Problem off = exponential_problem(t, exp_data(t, a, c));
    off.observations(3, 0) += .13; off.observations(7, 0) -= .09;
    Vector x(2); x << 1.8, 5.5;
    const auto exact_evaluation = varpro::evaluate(off, x, JacobianMode::exact);
    const auto kaufman_evaluation = varpro::evaluate(off, x);
    check((exact_evaluation.jacobian - kaufman_evaluation.jacobian).norm() > 1e-8,
          "exact mode changes the nonzero-residual Jacobian");
    check_residual_jacobian(off, x, linear_options, JacobianMode::exact, 3e-7,
                            "exact nonzero-residual Jacobian finite difference");
}

void test_rank_cases() {
    Vector t(6); for (Eigen::Index i = 0; i < t.size(); ++i) t(i) = static_cast<double>(i);
    Vector x(1); x << 2.0;
    Problem deficient;
    deficient.observations.resize(6, 1);
    for (Eigen::Index i = 0; i < 6; ++i) deficient.observations(i, 0) = 5.0 * std::exp(-t(i) / 2.0) + (i == 1 ? .2 : 0.0);
    deficient.basis = [t](const Vector& a) { Matrix b(t.size(), 2); for (Eigen::Index i = 0; i < t.size(); ++i) b.row(i) << std::exp(-t(i) / a(0)), std::exp(-t(i) / a(0)); return b; };
    deficient.derivative = [t](const Vector& a, Eigen::Index k) { Matrix d = Matrix::Zero(t.size(), 2); if (k == 0) for (Eigen::Index i = 0; i < t.size(); ++i) d.row(i) << std::exp(-t(i) / a(0)) * t(i) / (a(0) * a(0)), std::exp(-t(i) / a(0)) * t(i) / (a(0) * a(0)); return d; };
    auto e = varpro::evaluate(deficient, x);
    check(e.rank == 1, "deficient rank");
    close(e.coefficients(0, 0), e.coefficients(1, 0), 2e-12, "minimum-norm duplicated coefficients");
    Matrix d = deficient.derivative(x, 0), phi = deficient.basis(x);
    const double expected_sum = phi.col(0).dot(deficient.observations.col(0)) /
                                phi.col(0).squaredNorm();
    close(e.coefficients.col(0).sum(), expected_sum, 2e-10, "minimum-norm coefficient sum");
    close(e.coefficients(0, 0), expected_sum / 2.0, 2e-10, "minimum-norm coefficient value");
    Vector u = phi.col(0);
    u.normalize();
    Vector dc = (d * e.coefficients).col(0);
    const double projection_scale = u.dot(dc);
    Vector expected(6);
    for (Eigen::Index i = 0; i < expected.size(); ++i)
        expected(i) = projection_scale * u(i) - dc(i);
    close_matrix(e.jacobian, expected, 2e-10, "deficient range projector");

    Problem zero = deficient;
    zero.basis = [](const Vector&) { return Matrix::Zero(6, 2); };
    zero.derivative = [](const Vector&, Eigen::Index) { return Matrix::Zero(6, 2); };
    auto z = varpro::evaluate(zero, x);
    check(z.rank == 0, "rank zero");
    close_matrix(z.coefficients, Matrix::Zero(2, 1), 0.0, "rank zero coefficients");
    close_matrix(z.jacobian, Matrix::Zero(6, 1), 0.0, "rank zero Jacobian");
}

void test_scale_safe_rank_cutoff() {
    Problem p;
    p.observations.resize(2, 1);
    p.observations << 1.0, 0.0;
    p.basis = [](const Vector&) { Matrix b(2, 1); b << 1e308, 0.0; return b; };
    p.derivative = [](const Vector&, Eigen::Index) { return Matrix::Zero(2, 1); };
    Vector parameters(1); parameters << 1.0;
    auto e = varpro::evaluate(p, parameters);
    check(e.rank == 1, "large-scale basis rank cutoff");
    close(e.coefficients(0, 0) * 1e308, 1.0, 1e-12, "large-scale minimum-norm solve");
    close_matrix(e.residuals, Matrix::Zero(2, 1), 1e-12, "large-scale residual");
}

void test_configurable_rank_cutoff() {
    Problem p;
    p.observations.resize(2, 1);
    p.observations << 0.0, 1.0;
    p.basis = [](const Vector&) {
        Matrix b(2, 2);
        b << 1.0, 0.0, 0.0, 1e-8;
        return b;
    };
    p.derivative = [](const Vector&, Eigen::Index) { return Matrix::Zero(2, 2); };
    Vector parameters(1); parameters << 1.0;

    const auto automatic = varpro::evaluate(p, parameters);
    check(automatic.rank == 2, "automatic rank retains small singular value");
    close(automatic.coefficients(1, 0) * 1e-8, 1.0, 1e-12,
          "automatic rank solves small singular direction");

    LinearOptions truncated_options;
    truncated_options.rcond = 1e-6;
    const auto truncated = varpro::evaluate(p, parameters, truncated_options);
    check(truncated.rank == 1, "configured rank cutoff truncates small singular value");
    close_matrix(truncated.coefficients, Matrix::Zero(2, 1), 0.0,
                 "configured rank cutoff minimum-norm coefficients");
    Matrix expected_residuals(2, 1); expected_residuals << 0.0, 1.0;
    close_matrix(truncated.residuals, expected_residuals, 1e-12,
                 "configured rank cutoff residual");
}

template <class F> void expect_invalid(F&& f, const std::string& name) {
    try { f(); } catch (const std::invalid_argument&) { return; }
    throw std::runtime_error("expected invalid_argument: " + name);
}
template <class F> void expect_domain(F&& f, const std::string& name) {
    try { f(); } catch (const std::domain_error&) { return; }
    throw std::runtime_error("expected domain_error: " + name);
}

void test_errors_and_limit() {
    Vector t(5); for (Eigen::Index i = 0; i < 5; ++i) t(i) = static_cast<double>(i);
    Vector truth(2); truth << 1.0, 3.0;
    Matrix truth_coefficients(3, 1); truth_coefficients << 4.0, 2.5, 1.0;
    Matrix y = exp_data(t, truth, truth_coefficients);
    Problem p = exponential_problem(t, y);
    Vector a(2); a << 2.0, 6.5;
    Problem missing = p; missing.basis = {};
    expect_invalid([&] { varpro::evaluate(missing, a); }, "missing callback");
    Problem missing_derivative = p; missing_derivative.derivative = {};
    expect_invalid([&] { varpro::evaluate(missing_derivative, a); }, "missing derivative callback");
    Problem bad_data = p; bad_data.observations.resize(0, 1);
    expect_invalid([&] { varpro::evaluate(bad_data, a); }, "empty observations");
    bad_data = p; bad_data.observations(0, 0) = std::numeric_limits<double>::quiet_NaN();
    expect_invalid([&] { varpro::evaluate(bad_data, a); }, "nonfinite observations");
    Problem bad_shape = p; bad_shape.basis = [](const Vector&) { return Matrix::Ones(4, 3); };
    expect_invalid([&] { varpro::evaluate(bad_shape, a); }, "basis shape");
    Problem bad_weight = p; bad_weight.weights = Vector::Zero(5);
    expect_invalid([&] { varpro::evaluate(bad_weight, a); }, "all-zero weights");
    bad_weight.weights = Vector::Constant(5, -1.); expect_invalid([&] { varpro::evaluate(bad_weight, a); }, "negative weights");
    bad_weight.weights = Vector::Constant(5, std::numeric_limits<double>::quiet_NaN());
    expect_invalid([&] { varpro::evaluate(bad_weight, a); }, "nonfinite weights");
    Vector nan = a; nan(0) = std::numeric_limits<double>::quiet_NaN();
    expect_invalid([&] { varpro::evaluate(p, nan); }, "nonfinite parameters");
    Problem nan_basis = p; nan_basis.basis = [](const Vector&) { return Matrix::Constant(5, 3, std::numeric_limits<double>::infinity()); };
    expect_domain([&] { varpro::evaluate(nan_basis, a); }, "nonfinite basis");
    Problem nan_derivative = p; nan_derivative.derivative = [](const Vector&, Eigen::Index) { return Matrix::Constant(5, 3, std::numeric_limits<double>::quiet_NaN()); };
    expect_domain([&] { varpro::evaluate(nan_derivative, a); }, "nonfinite derivative");
    varpro::Options invalid_options; invalid_options.max_evaluations = 0;
    expect_invalid([&] { varpro::fit(p, a, invalid_options); }, "invalid options");
    invalid_options = {}; invalid_options.ftol = -1.0;
    expect_invalid([&] { varpro::fit(p, a, invalid_options); }, "negative tolerance");
    invalid_options = {};
    invalid_options.linear_options.rcond = std::numeric_limits<double>::quiet_NaN();
    expect_invalid([&] { varpro::fit(p, a, invalid_options); }, "nonfinite rank cutoff");
    invalid_options = {};
    invalid_options.jacobian_mode = static_cast<JacobianMode>(99);
    expect_invalid([&] { varpro::fit(p, a, invalid_options); }, "invalid Jacobian mode");

    varpro::Options limited; limited.max_evaluations = 1;
    auto r = varpro::fit(p, a, limited);
    check(r.status == varpro::Status::evaluation_limit, "evaluation-limit status");
    check(r.function_evaluations == 1, "evaluation-limit budget");
    close_matrix(r.parameters, a, 0.0, "evaluation-limit parameters");
    auto final_eval = varpro::evaluate(p, r.parameters);
    close_matrix(r.evaluation.coefficients, final_eval.coefficients, 1e-12, "final coefficients consistency");
    close_matrix(r.evaluation.residuals, final_eval.residuals, 1e-12, "final residual consistency");
}

} // namespace

int main() {
    const std::vector<std::pair<const char*, std::function<void()>>> tests = {
        {"single and MRHS fit", test_single_and_mrhs_fit},
        {"weighting and orthogonality", test_weighting_and_orthogonality},
        {"Kaufman Jacobians", test_kaufman_jacobians},
        {"exact Jacobians", test_exact_jacobians},
        {"rank deficient and rank zero", test_rank_cases},
        {"scale-safe rank cutoff", test_scale_safe_rank_cutoff},
        {"configurable rank cutoff", test_configurable_rank_cutoff},
        {"errors and evaluation limit", test_errors_and_limit},
    };
    int failures = 0;
    for (const auto& test : tests) {
        try { test.second(); std::cout << "[PASS] " << test.first << '\n'; }
        catch (const std::exception& e) { ++failures; std::cerr << "[FAIL] " << test.first << ": " << e.what() << '\n'; }
    }
    return failures == 0 ? 0 : 1;
}
