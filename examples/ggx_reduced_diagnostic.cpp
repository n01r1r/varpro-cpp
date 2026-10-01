#include "ggx_reduced_diagnostic.hpp"

#include <iomanip>
#include <iostream>

int main() {
  using namespace ggx_example;
  const auto g = fixture_geometry();
  Vector truth(3), trial(3);
  truth << std::log(0.25), std::log(0.55), 0.3;
  trial << std::log(0.30), std::log(0.50), 0.4;
  Vector y = 0.4 * g.a + specular(g, truth).s;
  Vector weights(y.size());
  for (Eigen::Index i = 0; i < y.size(); ++i) {
    y(i) += 1e-3 * std::sin(1.7 * static_cast<double>(i));
    weights(i) = 1.0 + 0.1 * (i % 3);
  }
  const auto reduced = evaluate(g, y, weights, trial, 0.2, 0.8);
  const auto local = spectrum(reduced.jacobian, 1e-10);
  std::cout << std::setprecision(12)
            << "coordinates: log(alpha_t), log(alpha_b), angle (radians)\n"
            << "mu_free / mu: " << reduced.mu_free << " / " << reduced.mu << '\n'
            << "linear Wa rank / retained condition: " << reduced.linear_rank
            << " / " << reduced.linear_diagnostics.condition_estimate << '\n'
            << "J_red singular values: "
            << local.diagnostics.singular_values.transpose() << '\n'
            << "J_red rank / retained condition (rcond=1e-10): " << local.rank
            << " / " << local.diagnostics.condition_estimate << '\n'
            << "weighted squared residual: " << reduced.residual.squaredNorm()
            << '\n';
  Vector isotropic(3);
  isotropic << std::log(0.4), std::log(0.4), 0.4;
  const auto iso = evaluate(g, y, weights, isotropic, 0.2, 0.8);
  const auto iso_local = spectrum(iso.jacobian, 1e-10);
  std::cout << "isotropic J_red singular values: "
            << iso_local.diagnostics.singular_values.transpose() << '\n'
            << "isotropic rank: " << iso_local.rank
            << " (angle is unobservable; retained condition excludes it)\n";
  return reduced.at_switch || local.rank != 3 || iso_local.rank != 2 ? 1 : 0;
}
