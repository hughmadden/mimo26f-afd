// Host-only parity comparison. Nonfinite expected OR actual values fail loud.
#pragma once
#include <cmath>
#include <limits>
#include <vector>

namespace m26_parity {
inline bool within_tol(const std::vector<float>& got, const std::vector<float>& expected,
                       double abs_tol, double rel_tol, double* worst) {
  *worst = std::numeric_limits<double>::infinity();
  if (got.empty() || got.size() != expected.size() ||
      !std::isfinite(abs_tol) || !std::isfinite(rel_tol) || abs_tol < 0 || rel_tol < 0)
    return false;
  double difference = 0, base = 0;
  for (size_t i = 0; i < got.size(); ++i) {
    if (!std::isfinite(got[i]) || !std::isfinite(expected[i])) return false;
    difference = std::fmax(difference, std::abs(double(got[i]) - double(expected[i])));
    base = std::fmax(base, std::abs(double(expected[i])));
  }
  *worst = difference;
  double bound = abs_tol + rel_tol * base;
  return std::isfinite(bound) && difference <= bound;
}
} // namespace m26_parity
