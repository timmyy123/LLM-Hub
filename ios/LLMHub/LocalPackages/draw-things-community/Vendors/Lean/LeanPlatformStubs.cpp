#include <lean/lean.h>

#include "runtime/io.h"

namespace lean {

void initialize_openssl() {}
void finalize_openssl() {}

}  // namespace lean

extern "C" LEAN_EXPORT lean_obj_res lean_openssl_version(lean_obj_arg) {
  return lean_box(0);
}

extern "C" LEAN_EXPORT lean_obj_res lean_io_exit(uint8_t) {
  return lean::io_result_mk_error(
      "IO.Process.exit is unavailable in embedded Lean");
}

extern "C" LEAN_EXPORT lean_obj_res lean_io_force_exit(uint8_t) {
  return lean::io_result_mk_error(
      "IO.Process.forceExit is unavailable in embedded Lean");
}
