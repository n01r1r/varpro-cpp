#include <cmath>
#include <iomanip>
#include <iostream>
#include <varpro/varpro.hpp>

int main() {
  constexpr varpro::Index m = 21;
  varpro::Vector t(m);
  for (varpro::Index i = 0; i < m; ++i) t(i) = static_cast<double>(i) * .5;

  varpro::Matrix observations(m, 2);
  for (varpro::Index i = 0; i < m; ++i) {
    const double slow = std::exp(-t(i) / 3.0);
    const double fast = std::exp(-t(i) / 1.0);
    observations(i, 0) = 4.0 * fast + 2.5 * slow + 1.0;
    observations(i, 1) = 1.5 * fast - 2.0 * slow + .25;
  }

  varpro::Problem problem;
  problem.observations = observations;
  problem.basis = [t](const varpro::Vector& a) {
    varpro::Matrix phi(t.size(), 3);
    for (varpro::Index i = 0; i < t.size(); ++i) {
      phi(i, 0) = std::exp(-t(i) / a(0));
      phi(i, 1) = std::exp(-t(i) / a(1));
      phi(i, 2) = 1.0;
    }
    return phi;
  };
  problem.derivative = [t](const varpro::Vector& a, varpro::Index k) {
    varpro::Matrix d = varpro::Matrix::Zero(t.size(), 3);
    if (k < 2)
      for (varpro::Index i = 0; i < t.size(); ++i)
        d(i, k) = std::exp(-t(i) / a(k)) * t(i) / (a(k) * a(k));
    return d;
  };

  varpro::Vector initial(2);
  initial << 1.8, 6.0;
  const varpro::FitResult result = varpro::fit(problem, initial);
  std::cout << std::setprecision(12) << "status: "
            << (result.converged() ? "converged" : "not converged") << '\n'
            << "parameters: " << result.parameters(0) << ' ' << result.parameters(1) << '\n'
            << "coefficients:\n";
  for (varpro::Index i = 0; i < result.evaluation.coefficients.rows(); ++i) {
    for (varpro::Index j = 0; j < result.evaluation.coefficients.cols(); ++j)
      std::cout << result.evaluation.coefficients(i,j) << ' ';
    std::cout << '\n';
  }
  std::cout << "squared error: " << result.evaluation.squared_error() << '\n';
  return result.converged() ? 0 : 1;
}
