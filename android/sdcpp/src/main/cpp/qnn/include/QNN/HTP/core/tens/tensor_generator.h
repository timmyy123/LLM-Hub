// ==============================================================================
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear
//
// ==============================================================================
#ifndef HEXNN_TENSOR_GENERATOR_H
#define HEXNN_TENSOR_GENERATOR_H 1
#include "dtype_enum.h"
#include "interface_defs.h"
#include "macros_attribute.h"
#include "weak_linkage.h"
#include "tensor_base.h"
#include "tensor_definitions.h"

#include <array>
#include <cstddef>
#include <memory>
#include <utility>

PUSH_VISIBILITY(default)

class Op;
class Graph;

namespace hnnx {

////// Tensor Generator //////////////

template <typename T, typename TX>
API_EXPORT inline std::unique_ptr<T> make_tensor_template(Op const *op, OutputDef const &odef, Graph &g)
{
    return std::unique_ptr<T>(std::make_unique<TX>(op, odef, g));
}

// we make tables of these entries:
//  rank, dtype, pointer to function which makes it.
// The tables are built only as static constexpr variable in tensor_generator_lookup<T>::lookup
// so there should be only one table per TensorType after link.
//
struct tensor_generator_table_entry {
    using T = Tensor; // maybe needs to be a template parm
    using maketens_funcp = std::unique_ptr<T> (*)(Op const *, OutputDef const &, Graph &);

    int rank;
    DType dtype;
    maketens_funcp fp;

    // default ctor
    inline constexpr tensor_generator_table_entry() : rank(), dtype(), fp() {}

    // each entry is constructed based on pointer to the tensor type.
    template <typename TX>
    inline constexpr tensor_generator_table_entry(TX const *)
        : rank(tensor_traits<TX>::rank), dtype(tensor_traits<TX>::dtype), fp(make_tensor_template<T, TX>)
    {
    }
};
// a thing to make the constexpr table..
template <typename TTUPLE, size_t... I>
inline constexpr std::array<tensor_generator_table_entry, std::tuple_size_v<TTUPLE>>
make_tengen_init(std::index_sequence<I...>)
{
    return {tensor_generator_table_entry(static_cast<typename std::tuple_element_t<I, TTUPLE> *>(nullptr))...};
}

template <typename TensorType> struct API_EXPORT tensor_generator_lookup {
    template <typename TX>
    using has_TensorType_as_base = std::integral_constant<bool, std::is_base_of<TensorType, TX>::value>;
    // this is a tuple of types for which T is a common base.
    using applicable_types = TupFilter_t<has_TensorType_as_base, TypicalTensors>;
    static constexpr size_t NTYPES = std::tuple_size_v<applicable_types>;

    static tensor_generator_table_entry const *lookup(int rank, DType dtype)
    {
        // this is a table of their rank, dtype, ctor function.
        static constexpr std::array<tensor_generator_table_entry, NTYPES> typedescs =
                make_tengen_init<applicable_types>(std::make_index_sequence<NTYPES>{});
        tensor_generator_table_entry const *p = typedescs.data();

        for (int i = 0; i < int(NTYPES); i++) {
            if (p->dtype == dtype && p->rank == rank) return p;
            p++;
        }
        return nullptr;
    }

    static std::unique_ptr<Tensor> make [[gnu::noinline]] (const Op *producer_in, const OutputDef &def, Graph &graph_in)
    {
        // concrete types get a shortcut if the dtype & rank match...
        if constexpr (!std::is_abstract<TensorType>::value) {
            if (def.dtype == tensor_traits<TensorType>::dtype && def.rank == tensor_traits<TensorType>::rank) {
                return make_tensor_template<Tensor, TensorType>(producer_in, def, graph_in);
            }
        }
        tensor_generator_table_entry const *const lookup_result = lookup(def.rank, def.dtype);
        if (lookup_result != nullptr) {
            return lookup_result->fp(producer_in, def, graph_in);
        }
        errlog("Lookup in %d tensor types failed", int(NTYPES));
        return nullptr;
    }
    // return true if 'make' would succeed.
    static bool is_valid(const OutputDef &def)
    {
        if constexpr (!std::is_abstract<TensorType>::value) {
            if (def.dtype == tensor_traits<TensorType>::dtype && def.rank == tensor_traits<TensorType>::rank)
                return true;
            else {
                debuglog(
                        "def.dtype %u, tensor_traits<TensorType>::dtype %u, def.rank %u, tensor_traits<TensorType>::rank %u",
                        (unsigned)def.dtype, (unsigned)tensor_traits<TensorType>::dtype, def.rank,
                        unsigned(tensor_traits<TensorType>::rank));
            }
        }
        return lookup(def.rank, def.dtype) != nullptr;
    }
};
// external API of tensor generator:
//    tensor_generator<T>( Op const *, OutputDef const &, Graph &) ->  std::unique_ptr<Tensor>
//    tensor_generator_valid<T>( Op const *, OutputDef const &, Graph &) ->  bool
//
// A call to tensor_generator<T>(..) is really a call to tensor_generator_lookup<T>::make(..)
//

API_EXPORT bool tensor_tall_crouton_disabled(Graph const &g);
API_EXPORT bool tensor_wide_crouton_disabled(Graph const &g);

template <typename T, typename = void> struct is_wide_crouton {
    static constexpr bool value = false;
};
template <typename T, typename = void> struct is_tall_crouton {
    static constexpr bool value = false;
};
template <typename T> struct is_wide_crouton<T, std::void_t<decltype(T::layout)>> {
    static constexpr bool value = (T::layout.chunk_total == 8 * 8 * 32) && (T::layout.ChunkSizes[2] > 1) &&
                                  (T::layout.ChunkSizes[1] < T::layout.ChunkSizes[2]);
};
template <typename T> struct is_tall_crouton<T, std::void_t<decltype(T::layout)>> {
    static constexpr bool value = (T::layout.chunk_total == 8 * 8 * 32) && (T::layout.ChunkSizes[1] > 1) &&
                                  (T::layout.ChunkSizes[1] >= T::layout.ChunkSizes[2]);
};

template <typename TensorType>
constexpr std::unique_ptr<Tensor> (*tensor_generator)(const Op *producer_in, const OutputDef &def,
                                                      Graph &graph_in) = tensor_generator_lookup<TensorType>::make;
template <typename TensorType>
API_FUNC_EXPORT inline bool tensor_generator_valid(const Op * /* producer_in */, const OutputDef &def, Graph &graph_in)
{
    if constexpr (is_wide_crouton<TensorType>::value) {
        using LType = typename tensor_traits<TensorType>::layouttensor_type;
        if (tensor_wide_crouton_disabled(graph_in) && tensor_traits<LType>::is_indirect) {
            debuglog("Wide croutons disabled...");
            return false;
        }
    }
    if constexpr (is_tall_crouton<TensorType>::value) {
        using LType = typename tensor_traits<TensorType>::layouttensor_type;
        if (tensor_tall_crouton_disabled(graph_in) && tensor_traits<LType>::is_indirect) {
            debuglog("Tall croutons disabled...");
            return false;
        }
    }
    return tensor_generator_lookup<TensorType>::is_valid(def);
}

// make a scalar tensor for a given def (with 0 rank, and specific dtype). Returns an empty
// pointer if there is no support.
API_FUNC_EXPORT std::unique_ptr<Tensor> tensor_generator_scalar(const Op *producer_in, const OutputDef &def,
                                                                void const *data, size_t len);

template <int relative_tolerance = 1 /* 1% */, int absolute_tolerance = 1 /* in 'FLT_EPSILON' ref <climits> */>
static inline constexpr int almost_eq(float rhs, float lhs)
{
    return std::abs(rhs - lhs) <= (
                                          // should it be max of (absolute, relative) ?
                                          (absolute_tolerance * std::numeric_limits<float>::epsilon()) +
                                          (relative_tolerance / 100.f * std::abs(lhs)));
}

using cmp_function = std::function<int(float, float)>;
#ifndef PREPARE_DISABLED
extern GraphStatus tensor_copy(Tensor &lhs, const Tensor &rhs);
#endif
extern GraphStatus check_dims(const Tensor &lhs, const Tensor &rhs);

//
// Set the shape of Tensor D to the same as Tensor S, and
// then copy the contents, adapting to whatever shapes and data format
//
API_FUNC_EXPORT void tensor_copy_4d(Tensor &dst, Tensor const &src);

API_FUNC_EXPORT void tensor_registry_testing();

template <typename T> struct memclass_of {
    static constexpr MemoryClass memclass = tensor_traits<T>::memclass;
};
template <> struct memclass_of<Tensor> {
    static constexpr MemoryClass memclass = MemoryClass::Default;
};

template <typename T> struct memclass_of<Vector<T *>> {
    static constexpr MemoryClass memclass = memclass_of<T>::memclass;
};
template <typename T> struct memclass_of<const Vector<T *>> {
    static constexpr MemoryClass memclass = memclass_of<T>::memclass;
};

template <MemoryClass C, typename... Ts> struct has_memclass;

template <MemoryClass C> struct has_memclass<C, std::tuple<>> {
    static constexpr bool value = false;
};

template <MemoryClass C, typename T, typename... Ts> struct has_memclass<C, std::tuple<T, Ts...>> {
    static constexpr bool value = memclass_of<T>::memclass == C || has_memclass<C, std::tuple<Ts...>>::value;
};

///////////////////////////////////////

// mechanism to generate functions to register tensor for serializing
// reg_tens_for_deser<T1,...>::f() -> int is a static function which
// registers T1,T2.
// To keep code size down, the code is only there for <T>; for more than
// one, the others are all called in sequence.
//
//  reg_tens_for_deser<T1,...>::f_ptr() is a static inline which returns
//  a pointer to f.
//
template <typename... T> struct reg_tens_for_deser {
    static int f() { return (reg_tens_for_deser<T>::f(), ...); }
    static constexpr auto f_ptr() -> int (*)() { return &f; }
};
// For empty list make fptr return null
// since most of the Ops have nothing to do, and a null pointer takes
// less code to make than the address of a function.
template <> struct reg_tens_for_deser<> {
    static constexpr auto f_ptr() -> int (*)() { return nullptr; }
};

// single item
//
template <typename T> struct reg_tens_for_deser<T> {
    static int f()
    {
        using TT = std::remove_reference_t<std::remove_cv_t<T>>;
        static_assert(std::is_same_v<T, TT>);
        if constexpr (!(std::is_abstract<T>::value)) {
            deserialize_tensor_register(typeid(T), type_name<T>(),
                                        deserialize_tensor_using_constructor<T>::deserialize);
        }
        return 0;
    }
    static constexpr auto f_ptr() -> int (*)()
    {
        if constexpr (!(std::is_abstract<T>::value)) {
            return &f;
        } else {
            // let's just have one empty function
            return reg_tens_for_deser<>::f_ptr();
        }
    }
};

template <typename TUP> struct map_rtfd_type {};
template <typename... T> struct map_rtfd_type<std::tuple<T...>> {
    using type = reg_tens_for_deser<T...>;
};
// given a tuple TUP, deserialize_tensor_tuple<TUP,FORCE>::f_ptr()
// returns a pointer to a function
// which registers all of the types which are not in SkipRegTensors<FORCE>
// (i.e. if FORCE is false, all of the types which are not in CoreTensors;
// if force is true, all of the types).
//

// This is SkipRegTensors<True>, which is used for FORCE=true (don't skip).
// SkipRegTensors<false> is defined at the bottom of tensors.h.
template <bool FORCE> struct SkipRegTensors {
    using type = std::tuple<>;
};

template <typename TUP, bool FORCE = false> struct deserialize_tensor_tuple {
    template <typename T> using not_core_tensor = not_contains_type<typename SkipRegTensors<FORCE>::type, T>;
    using filtered_T = std::conditional_t<FORCE, TUP, typename TupFilter<not_core_tensor, TUP>::type>;
    using rtfd_type = typename map_rtfd_type<filtered_T>::type;
    static constexpr auto f_ptr() -> int (*)() { return rtfd_type::f_ptr(); }
    static int f() { return rtfd_type::f(); }
};

template <> struct SkipRegTensors<false> {
    using type = CoreTensors;
};

} // namespace hnnx

POP_VISIBILITY()

#endif // HEXNN_TENSOR_GENERATOR_H
