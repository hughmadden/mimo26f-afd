// Host tests for the exact comparator used by the GPU parity executable.
#include "../../kernels/parity/compare.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

int main() {
  int count = 0;
  auto check = [&](bool ok, const char* name) {
    if (!ok) { std::fprintf(stderr, "FAIL %s\n", name); std::exit(1); }
    ++count;
  };
  double worst = 0;
  auto compare = [&](std::vector<float> a, std::vector<float> b,
                     double abs_tol = 1e-5, double rel_tol = 1e-5) {
    return m26_parity::within_tol(a, b, abs_tol, rel_tol, &worst);
  };
  float nan = std::numeric_limits<float>::quiet_NaN();
  float inf = std::numeric_limits<float>::infinity();
  check(compare({1,2}, {1,2}) && worst == 0, "exact finite match");
  check(compare({1}, {1.000001f}), "finite tolerance unchanged");
  check(!compare({1}, {2}), "finite mismatch");
  check(!compare({nan}, {1}) && std::isinf(worst), "NaN actual never passes");
  check(!compare({1}, {nan}), "NaN oracle never passes");
  check(!compare({nan}, {nan}), "NaN equality never passes");
  check(!compare({inf}, {inf}), "positive infinity equality never passes");
  check(!compare({-inf}, {-inf}), "negative infinity equality never passes");
  check(!compare({1, nan}, {1, 2}), "one unwritten lane never passes");
  check(!compare({}, {}), "empty result never passes");
  check(!compare({1}, {1, 2}), "shape mismatch never passes");
  check(!compare({1}, {1}, -1, 0), "negative absolute tolerance refuses");
  check(!compare({1}, {1}, 0, -1), "negative relative tolerance refuses");
  check(!compare({1}, {1}, nan, 0), "NaN tolerance refuses");
  check(!compare({1}, {1}, inf, 0), "infinite tolerance refuses");
  check(!compare({1}, {1}, 0, inf), "infinite relative tolerance refuses");
  check(!compare({2}, {2}, 0, std::numeric_limits<double>::max()), "overflowed bound refuses");
  float poison;
  std::memset(&poison, 0xff, sizeof(poison));
  check(std::isnan(poison) && !compare({poison}, {0}), "device output poison rejects skipped zero write");
  check(!compare({poison}, {42}), "device output poison rejects skipped nonzero write");
  // Reproduce the OLD fail-open max loop, proving the test detects the actual bug.
  double old_worst = 0;
  double difference = std::abs(double(poison) - 1.0);
  if (difference > old_worst) old_worst = difference;
  check(old_worst == 0 && !compare({poison}, {1}), "old comparator passes NaN; corrected comparator fails");
  std::printf("RESULT: PASS parity-selftest %d/%d\n", count, count);
}
