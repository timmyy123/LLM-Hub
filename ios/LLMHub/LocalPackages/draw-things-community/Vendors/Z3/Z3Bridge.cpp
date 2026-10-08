#include "Z3Bridge.h"
#include "Z3Embedded.h"

#include <cstdlib>
#include <cstring>
#include <exception>
#include <sstream>
#include <string>

#include "z3++.h"

namespace {

char *CopyString(const std::string &value) {
  char *copy = static_cast<char *>(std::malloc(value.size() + 1));
  if (copy == nullptr) {
    return nullptr;
  }
  std::memcpy(copy, value.c_str(), value.size() + 1);
  return copy;
}

Z3BridgeResult MakeResult(
    Z3BridgeStatus status, const std::string &model = {},
    const std::string &reason = {}) {
  return {
      status,
      model.empty() ? nullptr : CopyString(model),
      reason.empty() ? nullptr : CopyString(reason),
  };
}

std::string FormatModel(const z3::model &model) {
  std::ostringstream output;
  for (unsigned declarationIndex = 0; declarationIndex < model.size(); ++declarationIndex) {
    if (declarationIndex > 0) {
      output << '\n';
    }
    const z3::func_decl declaration = model[declarationIndex];
    output << declaration.name() << " = ";
    if (declaration.arity() == 0) {
      output << model.get_const_interp(declaration);
      continue;
    }
    const z3::func_interp interpretation = model.get_func_interp(declaration);
    output << '[';
    for (unsigned entryIndex = 0; entryIndex < interpretation.num_entries(); ++entryIndex) {
      if (entryIndex > 0) {
        output << ", ";
      }
      const z3::func_entry entry = interpretation.entry(entryIndex);
      output << '(';
      for (unsigned argumentIndex = 0; argumentIndex < entry.num_args(); ++argumentIndex) {
        if (argumentIndex > 0) {
          output << ", ";
        }
        output << entry.arg(argumentIndex);
      }
      output << ") -> " << entry.value();
    }
    if (interpretation.num_entries() > 0) {
      output << ", ";
    }
    output << "else -> " << interpretation.else_value() << ']';
  }
  return output.str();
}

}  // namespace

Z3BridgeResult Z3BridgeCheckSMT(const char *code, uint32_t timeoutMilliseconds) {
  if (code == nullptr) {
    return MakeResult(Z3BridgeStatusError, {}, "SMT-LIB input is null.");
  }
  std::lock_guard lock(z3_embedded::runtime_mutex());
  try {
    z3::context context;
    z3::solver solver(context);
    z3::params parameters(context);
    parameters.set("timeout", timeoutMilliseconds);
    solver.set(parameters);
    solver.from_string(code);
    switch (solver.check()) {
      case z3::sat:
        return MakeResult(Z3BridgeStatusSat, FormatModel(solver.get_model()));
      case z3::unsat:
        return MakeResult(Z3BridgeStatusUnsat);
      case z3::unknown: {
        const std::string reason = solver.reason_unknown();
        const bool timedOut = reason == "timeout" || reason == "canceled";
        return MakeResult(
            timedOut ? Z3BridgeStatusTimeout : Z3BridgeStatusUnknown, {}, reason);
      }
    }
  } catch (const z3::exception &error) {
    return MakeResult(Z3BridgeStatusError, {}, error.msg());
  } catch (const std::exception &error) {
    return MakeResult(Z3BridgeStatusError, {}, error.what());
  } catch (...) {
    return MakeResult(Z3BridgeStatusError, {}, "Unknown Z3 error.");
  }
  return MakeResult(Z3BridgeStatusUnknown, {}, "Z3 returned an invalid status.");
}

void Z3BridgeFreeResult(Z3BridgeResult result) {
  std::free(result.model);
  std::free(result.reason);
}
