#include <varpro/varpro.hpp>

// Compiled separately to check for non-inline definitions and ensure that
// no implementation object is needed to link the public header.
double header_only_other_translation_unit() {
  varpro::Problem p;
  p.observations = varpro::Matrix::Zero(2, 1);
  p.observations(0, 0) = 3.0;
  p.observations(1, 0) = 1.0;
  p.basis = [](const varpro::Vector&) {
    varpro::Matrix b(2, 1);
    b(0, 0) = 1.0;
    return b;
  };
  p.derivative = [](const varpro::Vector&, varpro::Index) {
    return varpro::Matrix::Zero(2, 1);
  };
  return varpro::evaluate(p, varpro::Vector{1.0}).squared_error();
}
