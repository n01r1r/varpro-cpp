#include <varpro/varpro.hpp>

int main() {
  varpro::Problem p;
  p.observations = varpro::Matrix::Ones(2, 1);
  p.basis = [](const varpro::Vector&) {
    return varpro::Matrix::Ones(2, 1);
  };
  p.derivative = [](const varpro::Vector&, varpro::Index) {
    return varpro::Matrix::Zero(2, 1);
  };
  const auto result = varpro::evaluate(p, varpro::Vector{1.0});
  return result.rank == 1 && result.squared_error() < 1e-20 ? 0 : 1;
}
