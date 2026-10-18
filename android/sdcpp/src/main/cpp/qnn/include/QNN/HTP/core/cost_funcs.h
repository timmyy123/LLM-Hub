// ==============================================================================
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear
//
// ==============================================================================

#ifndef COST_FUNCS_H
#define COST_FUNCS_H
#include <string_view>
#include <utility>
#include "weak_linkage.h"
#include "macros_attribute.h"
PUSH_VISIBILITY(default)

class Graph;
class Op;

namespace hnnx {

class API_EXPORT cost_function_t {
    using inner_func_t = float (*)(cost_function_t const &, const Graph &, Op const *);
    inner_func_t funcp;
    float val;

  public:
    cost_function_t(cost_function_t const &) = default;
    cost_function_t &operator=(cost_function_t const &) = default;
    constexpr explicit cost_function_t(float val_in) : funcp(simple_cost_function), val(val_in) {}
    constexpr cost_function_t(inner_func_t f, float val_in) : funcp(f), val(val_in) {}
    constexpr cost_function_t() noexcept : funcp(simple_cost_function), val(0.0f) {}

#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-equal"
#endif
    constexpr auto operator==(cost_function_t const &other) const noexcept
    {
        return (funcp == other.funcp) && (val == other.val);
    }
#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic pop // -Wfloat-equal
#endif

    constexpr auto operator!=(cost_function_t const &other) const noexcept { return !operator==(other); }

    inline float operator()(const Graph &graph_in, Op const *op) const { return (*funcp)(*this, graph_in, op); }
    static float simple_cost_function(cost_function_t const &self, const Graph &, Op const *)
    {
        return self.val;
    } // just returns val;

    constexpr auto get_val() const noexcept { return val; }

    // unreliable compare for two cost func: returns  -1,0,1 if this cost
    // is <,=,> than rhs cost, with the second result being true; or <0,false>
    // if it can't tell.
    std::pair<int, bool> compare(cost_function_t const &rhs) const;

    template <class T> static float cfunc(cost_function_t const &, const Graph &, Op const *);
};

API_EXPORT cost_function_t cost_func_from_str(std::string_view);

} // namespace hnnx

POP_VISIBILITY()

#endif
