// ==============================================================================
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear
//
// ==============================================================================

#ifndef OPS_OPTS_REGISTRATION_H
#define OPS_OPTS_REGISTRATION_H 1

// Note that Op files must include this file AFTER they've included either
// typical_op.h or variadic_op.h, since these headers both give definitions for DerivedType

#include "log.h"
#include "ops_opts_registration_defs.h"
#include "optimize_flags.h"
#include "optimize.h"
#include "op_register.h"
#include "op_register_ext.h"
#include "built_array.h"
#include "reg_op_node.h"
#include "reg_op_table.h"
#include "package_op_storage_base.h"
#include "reg_optim_node.h"

#include <cstdint>
#include <cinttypes>
#include <string>
#include <string_view>

namespace hnnx {
// LCOV_EXCL_START [SAFTYSWCCB-1736] constexprs resolved during compile time
/** @brief op_name_strtab_t empty struct to help specialize arr_container for the op_name string table */
struct op_name_strtab_t {};
/** @brief type_tag_strtab empty struct to help specialize arr_container for the type_tag string table */
struct type_tag_strtab_t {};

template <typename> constexpr bool is_strtab()
{
    return false;
}
template <> constexpr bool is_strtab<op_name_strtab_t>()
{
    return true;
}
template <> constexpr bool is_strtab<type_tag_strtab_t>()
{
    return true;
}
// LCOV_EXCL_STOP

/** @brief arr_container */
template <typename T, bool S = is_strtab<T>()> struct arr_container {
    /** @brief chain link to the built_array contained in this structure */
    template <typename UNIQ_TY, uint32_t I> static constexpr built_array<T, I> chain = {};
};

/** @brief arr_container<T, true> */
template <typename T> struct arr_container<T, true> {
    /** @brief chain link to the built_array contained in this structure */
    template <typename UNIQ_TY, uint32_t I, uint32_t S>
    static constexpr built_array<std::string::value_type, S> chain = {};
};

/** @brief reg_op_table_wrapper */
using reg_op_table_wrapper = reg_op_table const *(*)();

/** @brief reg_opt_table */
class reg_opt_table {
    reg_optim_node const *entries;
    uint32_t num_entries;
    std::string_view file_name;

  public:
    constexpr reg_optim_node const *get_entries() const noexcept { return entries; }
    constexpr uint32_t get_num_entries() const noexcept { return num_entries; }
    constexpr std::string_view const get_file_name() const noexcept { return file_name; }
    // LCOV_EXCL_START [SAFTYSWCCB-1736] constexprs resolved during compile time, consexpr constructor
    constexpr reg_opt_table(reg_optim_node const *const p, uint32_t const n,
                            std::string_view::value_type const *const f) noexcept
        : entries(p), num_entries(n), file_name{f}
    {
    }
    constexpr reg_opt_table() noexcept : reg_opt_table(nullptr, 0U, "") {}
    // LCOV_EXCL_STOP
};

/** @brief reg_opt_table_wrapper */
using reg_opt_table_wrapper = reg_opt_table const *(*)();

/** @brief op_name_strtab_container */
using op_name_strtab_container = arr_container<op_name_strtab_t>;
/** @brief type_tag_strtab_container */
using type_tag_strtab_container = arr_container<type_tag_strtab_t>;
/** @brief op_arr_container */
using op_arr_container = arr_container<reg_op_node>;
/** @brief opt_arr_container */
using opt_arr_container = arr_container<reg_optim_node>;
/** @brief op_table_arr_container */
using op_table_arr_container = arr_container<reg_op_table_wrapper>;
/** @brief opt_table_arr_container */
using opt_table_arr_container = arr_container<reg_opt_table_wrapper>;

/** @brief ba_str a built_array of char strings */
template <uint32_t I> using ba_str = built_array<std::string::value_type, I>;

/** @brief ba_op a built_array of reg_op_nodes */
template <uint32_t I> using ba_op = built_array<reg_op_node, I>;

/** @brief ba_opt a built_array of reg_optim_nodes */
template <uint32_t I> using ba_opt = built_array<reg_optim_node, I>;

/** @brief ba_op_table a built_array of reg_op_table_wrappers */
template <uint32_t I> using ba_op_table = built_array<reg_op_table_wrapper, I>;

/** @brief ba_opt_table a built_array of reg_opt_table_wrappers */
template <uint32_t I> using ba_opt_table = built_array<reg_opt_table_wrapper, I>;

/**
 * @brief
 * NodeCounter template converts the __COUNTER__'s
 * current value to counts for number of reg_op_nodes and
 * reg_optim_nodes created so far. It is specialized upon every
 * REGISTER_OP/DEF_OPT by incrementing either "reg_op_count"
 * or "reg_opt_count", with member functions that can get the
 * current counts.
 */
// LCOV_EXCL_START [SAFTYSWCCB-1736] constexprs resolved during compile time
template <typename UNIQ_TY, int32_t I> class NodeCounter {
  public:
    static constexpr std::string_view get_op() noexcept { return ""; }
    static constexpr std::string_view get_tag() noexcept { return ""; }

  private:
    /** @brief inc_op @return 0, or 1 if the op count is incremented */
    constexpr static int32_t inc_op() noexcept { return 0; }
    /** @brief inc_opt @return 0, or 1 if the opt count is incremented */
    constexpr static int32_t inc_opt() noexcept { return 0; }
    /** @brief inc_op_name_strtab_size @return 0, or some string size constant if the string table needs to grow */
    constexpr static uint64_t inc_op_name_strtab_size() noexcept { return 0; }
    /** @brief inc_type_tag_strtab_size @return 0, or some string size constant if the string table needs to grow */
    constexpr static uint64_t inc_type_tag_strtab_size() noexcept { return 0; }

  public:
    /** @brief reg_op_count @return The number of ops that have been registered so far */
    constexpr static int32_t reg_op_count() noexcept { return inc_op() + NodeCounter<UNIQ_TY, I - 1>::reg_op_count(); }
    /** @brief reg_op_count @return The number of opts that have been registered so far */
    constexpr static int32_t reg_opt_count() noexcept
    {
        return inc_opt() + NodeCounter<UNIQ_TY, I - 1>::reg_opt_count();
    }
    /** @brief op_name_strtab_size @return The string table size for the ops that have been registered so far */
    constexpr static uint64_t op_name_strtab_size() noexcept
    {
        return inc_op_name_strtab_size() + NodeCounter<UNIQ_TY, I - 1>::op_name_strtab_size();
    }
    /** @brief type_tag_strtab_size @return The string table size for the ops that have been registered so far */
    constexpr static uint64_t type_tag_strtab_size() noexcept
    {
        return inc_type_tag_strtab_size() + NodeCounter<UNIQ_TY, I - 1>::type_tag_strtab_size();
    }
};

/** @brief Shorthand for op_name_strtab_container::chain<...> */
template <typename U, size_t I> constexpr auto op_name_chain()
{
    return op_name_strtab_container::chain<U, NodeCounter<U, I>::reg_op_count(),
                                           NodeCounter<U, I>::op_name_strtab_size()>;
}

/** @brief Shorthand for type_tag_strtab_container::chain<...> */
template <typename U, size_t I> constexpr auto type_tag_chain()
{
    return type_tag_strtab_container::chain<U, NodeCounter<U, I>::reg_op_count(),
                                            NodeCounter<U, I>::type_tag_strtab_size()>;
}

/**
 * @brief StrtabUpdate A class template for storing:
 * - The result of the existence check
 * - The offset the new string
 * when appending to the Op name and type suffix string tables.
 *
 * @tparam U Unique type for identifying the translation unit containing this update
 * @tparam I Unique index number identifying the "ith" Op to be registered in this file
 */
template <typename U, uint32_t I> struct StrtabUpdate {
    /** @brief is_new_op_name whether the Op name to be appended is already present in the op_name_strtab */
    static bool const is_new_op_name;
    /** @brief op_name_offset short offset locating the Op name in the op_name_strtab */
    static uint16_t const op_name_offset;
    /** @brief is_new_type_tag whether the type suffix to be appended is already present in the type_tag_strtab */
    static bool const is_new_type_tag;
    /** @brief type_tag_offset short offset locating the type suffix in the type_tag_strtab */
    static uint16_t const type_tag_offset;
};

/**
 * @brief strtab_append Append string to table iff it is not already present.
 * @tparam U Unique type to ensure independence of specializations across translation units
 * @tparam I REGISTER_OP index number
 * @tparam N Current table size
 */
template <typename U, uint32_t I, std::string_view::size_type M, bool A, uint32_t N>
constexpr auto strtab_append(ba_str<N> const &curr, std::string_view const newString)
{
    if constexpr (A) {
        sv_size_wrapper<M> const w{newString};
        return curr.append(w);
    } else {
        return curr;
    }
}

/** @brief make_string_view Convert an array of string data into a string_view, but
 *  substitute in an empty string if the array's .data() would be nullptr.
 */
template <std::string_view::size_type N>
constexpr std::string_view make_string_view(std::array<std::string::value_type, N> const &arr) noexcept
{
    return arr.size() != 0 ? std::string_view{arr.data(), arr.size()} : std::string_view{"", 1};
}
// LCOV_EXCL_STOP

} // namespace hnnx

#endif // OPS_OPTS_REGISTRATION_H
