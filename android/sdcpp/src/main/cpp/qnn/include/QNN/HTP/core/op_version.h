// ==============================================================================
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear
//
// ==============================================================================

#ifndef OP_VERSION_H
#define OP_VERSION_H 1
#include "op_registry.h"

namespace hnnx {
PUSH_VISIBILITY(default)

struct OpVersion {
    enum OpVersionEnum { V0 = 0, V1, V2, V3, V4, V5, V6, V7, V8 };
    OpVersion(const OpVersion &) = delete; // copy constructor, required by static analyzer
    OpVersion(OpVersion &&) = delete; // move constructor, required by static analyzer
    OpVersion &operator=(const OpVersion &) = delete; // copy assignemnt, required by static analyzer
    OpVersion &operator=(OpVersion &&) = delete; // move assignment, required by static analyzer
    OpVersion(OpVersionEnum opver) : op_version(opver) {}
    int to_int() const { return static_cast<int>(op_version); }
    bool operator==(OpVersionEnum opver) const { return op_version == opver; }
    bool operator==(int opver) const { return static_cast<int>(op_version) == opver; }
    ~OpVersion() = default;

  private:
    OpVersionEnum op_version;
};

// Operation versioning extension strings. Convert OpVersion to string "^XX"
// IMPORTANT: Do NOT modify any existing version string literals to preserve backward compatibility
#define OpVersionExtV0 ""
#define OpVersionExtV1 "^010"
#define OpVersionExtV2 "^020"
#define OpVersionExtV3 "^030"
#define OpVersionExtV4 "^040"
#define OpVersionExtV5 "^050"
#define OpVersionExtV6 "^060"
#define OpVersionExtV7 "^070"
#define OpVersionExtV8 "^080"

// generate uint64_t
// [63 ........ 48][47 ........ 32][31 .......................... 0]
//     MAJOR           MINOR                   PATCH
#define MAKE_KERNEL_SW_VERSION(major, minor, patch)                                                                    \
    ((((uint64_t)(major) & 0xFFFFu) << 48u) | (((uint64_t)(minor) & 0xFFFFu) << 32u) |                                 \
     ((uint64_t)(patch) & 0xFFFFFFFFu))

#define GLOBAL_KERNEL_VERSION MAKE_KERNEL_SW_VERSION(0, hnnx::OpVersion::V5, 0)

} // namespace hnnx
POP_VISIBILITY()
#endif
