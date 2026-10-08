#include "Z3Bridge.h"

#include <cassert>
#include <string>

int main() {
  Z3BridgeResult sat = Z3BridgeCheckSMT(
      "(set-logic QF_LIA) (declare-const x Int) (assert (> x 4))", 5000);
  assert(sat.status == Z3BridgeStatusSat);
  assert(sat.model != nullptr);
  assert(std::string(sat.model).find("x") != std::string::npos);
  Z3BridgeFreeResult(sat);

  Z3BridgeResult unsat = Z3BridgeCheckSMT(
      "(set-logic QF_LIA) (declare-const x Int) (assert (> x 4)) "
      "(assert (< x 3))",
      5000);
  assert(unsat.status == Z3BridgeStatusUnsat);
  Z3BridgeFreeResult(unsat);

  Z3BridgeResult malformed = Z3BridgeCheckSMT("(assert", 5000);
  assert(malformed.status == Z3BridgeStatusError);
  assert(malformed.reason != nullptr);
  Z3BridgeFreeResult(malformed);
  return 0;
}
