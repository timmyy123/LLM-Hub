// ==============================================================================
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear
//
// ==============================================================================

#ifndef REG_OP_NODE_H_
#define REG_OP_NODE_H_

#include "op_reg_parms.h"
#include "make_op_custom.h"
#include "package_op_storage_base.h"
#include "op_register_ext_fwd.h"

#include <cstdint>
#include <string_view>
#include <memory>
#include <vector>

namespace hnnx {
class PackageOpStorageBase;
struct reg_op_node_probe;

/** @brief reg_op_node */
class reg_op_node {
    friend struct reg_op_node_probe;

    /** @brief parms parameters (cost func, flags, etc) for the Op */
    op_reg_parms op_parms;
    /** @brief op_name */
    std::uint16_t op_name_offset;
    /** @brief type_tag */
    std::uint16_t type_tag_offset;
    /** @brief line number where op is registered */
    [[maybe_unused]] std::uint16_t line;
    bool is_external;
    bool is_legacy;
    std::string_view target_reg;

    std::string_view const get_subview(std::string_view const strtab, std::string_view::size_type const start) const
    {
        return std::string_view{strtab.data() + start};
    }

  public:
    // LCOV_EXCL_START [SAFTYSWCCB-1736] constexprs resolved during compile time
    /** @brief reg_op_node @param p @param n @param t @param l @param s @param g @param reg */
    constexpr reg_op_node(op_reg_parms const p, std::uint16_t const n, std::uint16_t const t, std::uint16_t const l,
                          bool s, bool g, std::string_view const reg) noexcept
        : op_parms(p), op_name_offset(n), type_tag_offset(t), line(l), is_external(s), is_legacy(g), target_reg(reg)
    {
    }

    constexpr reg_op_node(op_reg_parms const p, std::uint16_t const n, std::uint16_t const t, std::uint16_t const l,
                          bool s, bool g) noexcept
        : op_parms(p), op_name_offset(n), type_tag_offset(t), line(l), is_external(s), is_legacy(g), target_reg("core")
    {
    }

    /** @brief reg_op_node */
    constexpr reg_op_node() noexcept : reg_op_node(op_reg_parms{}, 0, 0, 0, false, false, "core") {}

    // LCOV_EXCL_STOP

    /** @brief process invoke the make_op_custom function */
    void core_process(std::string_view const op_name_strtab, std::string_view const type_tag_strtab) const
    {
        std::string_view const op_name = get_subview(op_name_strtab, op_name_offset);
        std::string_view const type_tag = get_subview(type_tag_strtab, type_tag_offset);
        hnnx::make_op_custom(op_name, type_tag, op_parms, is_legacy);
    }

    /** @brief process append external oppkg ops into op vector for later use */
    void pkg_process(std::string_view const op_name_strtab, std::string_view const type_tag_strtab) const
    {
        std::string_view const op_name = get_subview(op_name_strtab, op_name_offset);
        std::string_view const type_tag = get_subview(type_tag_strtab, type_tag_offset);
        std::vector<std::unique_ptr<PackageOpStorageBase>> &ops = current_package_ops_storage_vec_func();
        // support typical ops package
        ops.push_back(std::make_unique<PackageOpStorageBase>(op_name, type_tag, is_external, op_parms, target_reg));
    }

    template <typename CB>
    void visit(std::string_view const op_name_strtab, std::string_view const type_tag_strtab,
               std::string_view const file_name, CB &&cb) const
    {
        std::string_view const op_name = get_subview(op_name_strtab, op_name_offset);
        std::string_view const type_tag = get_subview(type_tag_strtab, type_tag_offset);
        cb(op_name, type_tag, file_name, static_cast<int>(line));
    }
#ifndef PREPARE_DISABLED
    void core_process(std::string_view const op_name_strtab, std::string_view const type_tag_strtab,
                      std::string_view file_name) const
    {
        std::string_view const op_name = get_subview(op_name_strtab, op_name_offset);
        std::string_view const type_tag = get_subview(type_tag_strtab, type_tag_offset);
        hnnx::make_op_custom(op_name, type_tag, op_parms, is_legacy, file_name, line, target_reg);
    }
    void pkg_process(std::string_view const op_name_strtab, std::string_view const type_tag_strtab,
                     std::string_view /* file_name */) const
    {
        pkg_process(op_name_strtab, type_tag_strtab);
    }
#endif

    // LCOV_EXCL_START [SAFTYSWCCB-1736] constexprs resolved during compile time
    constexpr bool operator==(reg_op_node const &other) const noexcept
    {
        return (op_parms == other.op_parms) && (op_name_offset == other.op_name_offset) &&
               (type_tag_offset == other.type_tag_offset) && (line == other.line) &&
               (is_external == other.is_external) && (is_legacy == other.is_legacy) && (target_reg == other.target_reg);
    }
    constexpr bool operator!=(reg_op_node const &other) const noexcept { return !(operator==(other)); }
    // LCOV_EXCL_STOP
};
} // namespace hnnx
#endif
