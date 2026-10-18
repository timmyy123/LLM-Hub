// ==============================================================================
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear
//
// ==============================================================================

#ifndef REG_OPTIM_NODE_H_
#define REG_OPTIM_NODE_H_

#include "optimize_flags.h"
#include "optimize_fwd.h"
#include "entire_defopt.h"

#include <cstdint>
#include <string_view>

namespace hnnx {
#ifdef PREPARE_DISABLED
/** @brief reg_optim_node This is stub class that does not
 *  register DEF_OPTs when prepare is disabled.
*/
class reg_optim_node {
  public:
    /** @brief No-op when prepare is disabled */
    void core_process(std::string_view const fname) const { (void)fname; }

    /** @brief No-op when prepare is disabled */
    void pkg_process(std::string_view const fname) const { (void)fname; }
};
#else
/** @brief reg_optim_node */
class reg_optim_node {
    /** @brief defopt */
    hnnx::get_entire_defopt_t defopt;
    /** @brief flags */
    OptimFlags::flags_t flags;
    /** @brief priority */
    uint16_t priority;
    /** @brief line */
    uint16_t line;
    /** @brief target_registry */
    std::string_view target_reg;

  public:
    /** @brief reg_optim_node @param p @param fl @param m @param c @param r @param f @param l */
    constexpr reg_optim_node(uint16_t const p, OptimFlags::flags_t const fl, hnnx::get_entire_defopt_t d,
                             uint16_t const l) noexcept
        : defopt(d), flags(fl), priority(p), line(l), target_reg("core")
    {
    }

    constexpr reg_optim_node(uint16_t const p, OptimFlags::flags_t const fl, hnnx::get_entire_defopt_t d,
                             uint16_t const l, std::string_view const target_reg) noexcept
        : defopt(d), flags(fl), priority(p), line(l), target_reg(target_reg)
    {
    }

    /** @brief reg_optim_node */
    constexpr reg_optim_node() noexcept : reg_optim_node(0, 0U, nullptr, 0) {}

    /** @brief process invoke the add_package_opt function */
    void core_process(std::string_view const fname) const
    {
        hnnx::add_package_opt(current_package_opts_storage_vec_func(), priority, flags, defopt, fname.data(), line,
                              target_reg);
    }

    /** @brief process invoke the add_package_opt function for external oppkg */
    void pkg_process(std::string_view const fname) const
    {
        hnnx::add_package_opt(current_package_opts_storage_vec_func(), priority, flags, defopt, fname.data(), line,
                              target_reg);
    }
};
#endif

} // namespace hnnx
#endif
