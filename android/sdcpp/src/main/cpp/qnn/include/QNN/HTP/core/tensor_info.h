// ==============================================================================
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear
//
// ==============================================================================

#ifndef TENSOR_INFO_H
#define TENSOR_INFO_H 1

#include "opname_tag.h"
#include "bit_vector.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <string_view>
#include <utility>
#include <vector>

namespace hnnx {

struct TensorInfo {
    using TensorFlags = unsigned; // treatad a a bit vector
    using NameSet = BitVector<uint32_t>; // a bit vector of input and output names
    // outputs are the first "num_outputs" bits

    hnnx::opname_tag_t op_name;
    unsigned num_outputs = 1; // at least 1 "*" but can be longer

    NameSet fixed; // names which do not change, roughly flat&main_memory
    NameSet is_crouton;
    NameSet is_flat;
    NameSet is_main_memory;
    NameSet is_tcm;
    NameSet prefer_tcm; // place in tcm if they fit
    NameSet fixed_constants;
    NameSet unaligned_ok;
    NameSet crouton_if; // assign crouton if any input name in this set is

    std::vector<std::tuple<TensorFlags, hnnx::opname_tag_t>> renames;
    std::vector<std::pair<NameSet, opname_tag_t>> early_renames;
    // std::vector<std::pair<unsigned, Flags>> outputs;
};
} // namespace hnnx

namespace DefProperties {
enum class PropertyFlags {
    TCM = 1,
    MAIN_MEMORY = 2,
    CROUTON = 4,
    FLAT = 8,
    INVARIANT = 16,
};

constexpr PropertyFlags FLAT = PropertyFlags::FLAT;
constexpr PropertyFlags MAIN_MEMORY = PropertyFlags::MAIN_MEMORY;
constexpr PropertyFlags CROUTON = PropertyFlags::CROUTON;
constexpr PropertyFlags TCM = PropertyFlags::TCM;
constexpr PropertyFlags INVARIANT = PropertyFlags::INVARIANT;
constexpr unsigned OTHERWISE = 0;

// Rename(flag-list, new_name) if after migration all the specified
// flags are set, the operator is renamted to new_name
// Example: Rename(CROUTON,TCM, "Add.tcm")
struct Rename {
    unsigned flags = 0;
    const char *name = nullptr;
    void update() {}
    void update1(const PropertyFlags flag) { flags |= static_cast<unsigned>(flag); }
    void update1(const unsigned flag) { assert(flag == OTHERWISE); }
    void update1(const char *const name_p) { name = name_p; }
    template <typename First, typename... Rest> void update(First first, Rest... rest)
    {
        update1(first);
        update(rest...);
    }
    template <typename... Flags> explicit Rename(Flags... flags_) { update(flags_...); }
};

struct StringList {
    std::vector<const char *> names;
    void update() {}
    void update1(const char *const name) { names.push_back(name); }
    template <typename First, typename... Rest> void update(First first, Rest... rest)
    {
        update1(first);
        update(rest...);
    }
    template <typename... Args> explicit StringList(Args... args) { update(args...); }
};

struct Op : public StringList {
    template <typename... Args> explicit Op(Args... args) : StringList(args...) {}
};
struct Outputs : public StringList {
    template <typename... Args> explicit Outputs(Args... args) : StringList(args...) {}
};
struct Fixed : public StringList {
    template <typename... Args> explicit Fixed(Args... args) : StringList(args...) {}
};
struct Crouton : public StringList {
    template <typename... Args> explicit Crouton(Args... args) : StringList(args...) {}
};
struct Flat : public StringList {
    template <typename... Args> explicit Flat(Args... args) : StringList(args...) {}
};
struct MainMemory : public StringList {
    template <typename... Args> explicit MainMemory(Args... args) : StringList(args...) {}
};
struct Tcm : public StringList {
    template <typename... Args> explicit Tcm(Args... args) : StringList(args...) {}
};
struct PreferTcm : public StringList {
    template <typename... Args> explicit PreferTcm(Args... args) : StringList(args...) {}
};

struct FixedConstant : public StringList {
    template <typename... Args> explicit FixedConstant(Args... args) : StringList(args...) {}
};

struct UnalignedOk : public StringList {
    template <typename... Args> explicit UnalignedOk(Args... args) : StringList(args...) {}
};
struct WhenFitsIfTCM : public StringList {
    template <typename... Args> explicit WhenFitsIfTCM(Args... args) : StringList(args...) {}
};
struct CroutonIf : public StringList {
    template <typename... Args> explicit CroutonIf(Args... args) : StringList(args...) {}
};

struct TilingOnly {};

// An object that is used to build a TensorInfo from a variadic constructor clal
struct TensorInfoBuilder : public hnnx::TensorInfo {
    const char *package;
    std::vector<const char *> names;
    std::vector<const char *> output_names;
    bool tiling_only = false;
    // convert to flags (with ellipsis folded into set_at_and_above_pos)
    API_FUNC_EXPORT NameSet nameset(const StringList &list, bool input_only = false);
    void update() {}

    // Old style, just the root name
    void update1(const char *const name)
    {
        assert(names.empty());
        op_name = name;
        names.push_back("*");
    }
    // new style Op(root_name, ...)
    void update1(Op names_p)
    {
        assert(names.empty());
        names = std::move(names_p.names);
        assert((std::all_of(names.begin(), names.end(),
                            [](const char *name) { return std::strcmp(name, "...") != 0; })) &&
               "Invalid name '...'");
        op_name = names[0];
        names[0] = "*";
    }
    void update1(Outputs names_p)
    {
        assert(output_names.empty());
        assert(!names_p.names.empty());
        num_outputs = static_cast<unsigned>(names_p.names.size());
        output_names = std::move(names_p.names);
        assert((std::all_of(output_names.begin(), output_names.end(),
                            [](const char *name) { return std::strcmp(name, "...") != 0; })) &&
               "Invalid name '...'");
    }
    void update_flags(NameSet flags_, NameSet &true_, const NameSet false_)
    {
        flags_ &= ~(true_ | false_ | fixed); // don't change a position alreay set
        true_ |= flags_;
    }
    void update1(const Fixed fixed_p)
    {
        const NameSet fixed_names = nameset(fixed_p);
        const NameSet really_fixed = fixed_names & ~(is_flat | is_crouton | is_tcm | prefer_tcm | is_main_memory);
        update_flags(fixed_names, is_flat, is_crouton);
        update_flags(fixed_names, is_main_memory, is_tcm | prefer_tcm);
        fixed |= really_fixed;
    }
    void update1(const PreferTcm operand_list) { prefer_tcm |= nameset(operand_list, /*input_only*/ true); }
    void update1(const Flat operand_list) { update_flags(nameset(operand_list), is_flat, is_crouton); }
    void update1(const Crouton operand_list) { update_flags(nameset(operand_list), is_crouton, is_flat); }
    void update1(const MainMemory operand_list)
    {
        update_flags(nameset(operand_list), is_main_memory, is_tcm | prefer_tcm);
    }
    void update1(const Tcm operand_list) { update_flags(nameset(operand_list), is_tcm, is_main_memory); }
    void update1(const UnalignedOk operand_list) { unaligned_ok |= nameset(operand_list); }
    void update1(const FixedConstant operand_list) { fixed_constants |= nameset(operand_list, /*input_only*/ true); }
    void update1(const CroutonIf operand_list) { crouton_if |= nameset(operand_list, /*input_only*/ true); }
    void update1(const TilingOnly) { tiling_only = true; }
    API_FUNC_EXPORT void update1(Rename rename); // in tcm_migration.cc
    // void update1(Output output) { outputs.emplace_back(output.index, output.flags); }
    void update1(WhenFitsIfTCM rename);

    // void update1(NoPropagationToArgs stops_p) { stops |= stops_p.positions; }
    // void update1(NoPropagationAboveArg above) { stops |= 0xffff << (above.position + 1); }
    // void update1(const unsigned flag) { flags |= flag; }
    template <typename First, typename... Rest> void update(First first, Rest... rest)
    {
        update1(first);
        update(rest...);
        assert(!names.empty() && "No name specified for DEF_TENSOR_PROPERTIES");
    }

    template <typename... Args> TensorInfoBuilder(const char *package_, Args... args) : package(package_)
    {
        update(args...);
    }
    TensorInfoBuilder() = default;
};

} // namespace DefProperties

namespace hnnx {
// In tcm_migration.cc
extern API_FUNC_EXPORT bool register_tensor_properties(const char *package, DefProperties::TensorInfoBuilder);
// Registry-scoped variant
extern API_FUNC_EXPORT bool register_tensor_properties(const char *package, DefProperties::TensorInfoBuilder,
                                                       std::string_view registry);
} // namespace hnnx

#endif // TENSOR_INFO_H
