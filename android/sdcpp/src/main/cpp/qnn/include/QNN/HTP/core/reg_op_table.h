// ==============================================================================
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear
//
// ==============================================================================
#ifndef HEXNN_PUB_REG_OP_TABLE_H
#define HEXNN_PUB_REG_OP_TABLE_H

#include "reg_op_node.h"

#include <cstdint>
#include <string_view>

namespace hnnx {

constexpr std::string_view op_file_name(std::string_view path)
{
#ifdef HEXNN_OP_FILE_FULL_PATH
    return path;
#else // #ifdef HEXNN_OP_FILE_FULL_PATH
    size_t last_pos = path.find_last_of('/');
    if (last_pos == std::string_view::npos) {
        return path;
    }
    size_t second_last_pos = path.find_last_of('/', last_pos - 1);
    if (second_last_pos == std::string_view::npos) {
        return path;
    }
    size_t third_last_pos = path.find_last_of('/', second_last_pos - 1);
    if (third_last_pos == std::string_view::npos) {
        return path;
    }
    return path.substr(third_last_pos + 1);
#endif // #else // #ifdef HEXNN_OP_FILE_FULL_PATH
}

/** @brief reg_op_table */
class reg_op_table {
    reg_op_node const *entries;
    uint32_t num_entries;
    std::string_view op_name_strtab;
    std::string_view type_tag_strtab;
    std::string_view file_name;

  public:
    constexpr reg_op_node const *get_entries() const noexcept { return entries; }
    constexpr uint32_t get_num_entries() const noexcept { return num_entries; }
    constexpr std::string_view const get_op_name_strtab() const noexcept { return op_name_strtab; }
    constexpr std::string_view const get_type_tag_strtab() const noexcept { return type_tag_strtab; }
    constexpr std::string_view const get_file_name() const noexcept { return file_name; }
    // LCOV_EXCL_START [SAFTYSWCCB-1736] constexprs resolved during compile time, consexpr constructor
    constexpr reg_op_table(reg_op_node const *const p, uint32_t const n, std::string_view::value_type const *const o,
                           std::string_view::size_type const o_size, std::string_view::value_type const *const t,
                           std::string_view::size_type const t_size,
                           std::string_view::value_type const *const f) noexcept
        : entries(p), num_entries(n), op_name_strtab{o, o_size}, type_tag_strtab{t, t_size}, file_name{op_file_name(f)}
    {
    }
    constexpr reg_op_table() noexcept : reg_op_table(nullptr, 0U, "", 0U, "", 0U, "") {}
    // LCOV_EXCL_STOP

    // LCOV_EXCL_START [SAFTYSWCCB-1736] constexprs resolved during compile time
    constexpr bool operator==(reg_op_table const &other) const noexcept
    {
        auto const num_entries_equal{num_entries == other.num_entries};
        if (!num_entries_equal) {
            return false;
        }

        auto const entries_is_nullptr{entries == nullptr};
        auto const other_entries_is_nullptr{other.entries == nullptr};
        if (entries_is_nullptr ^ other_entries_is_nullptr) {
            return false;
        }

        if (!entries_is_nullptr && !other_entries_is_nullptr) {
            auto const n{num_entries};

            for (std::size_t i{0}; i < n; i++) {
                if (entries[i] != other.entries[i]) {
                    return false;
                }
            }
        }

        auto const rest_equal{(op_name_strtab == other.op_name_strtab) && (type_tag_strtab == other.type_tag_strtab) &&
                              (file_name == other.file_name)};

        return rest_equal;
    }
    // LCOV_EXCL_STOP
};
} // namespace hnnx
#endif // HEXNN_PUB_REG_OP_TABLE_H
