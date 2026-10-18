// ==============================================================================
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear
//
// ==============================================================================
#ifndef HEXNN_TENSOR_DEFINITIONS_H
#define HEXNN_TENSOR_DEFINITIONS_H 1
#include "dtype.h"
#include "macros_attribute.h"
#include "memory_layout.h"
#include "padding.h"
#include "tensor_concrete.h"
#include "weak_linkage.h"

#include <cstdint>

PUSH_VISIBILITY(default)

namespace Ldefs {
template <unsigned elbytes> struct stype_for;
template <> struct stype_for<1> {
    using type = uint8_t;
};
template <> struct stype_for<2> {
    using type = uint16_t;
};
template <> struct stype_for<4> {
    using type = NN_UINT32_T;
};
template <> struct stype_for<8> {
    using type = NN_UINT64_T;
};
} // namespace Ldefs
// macro to define a layout config struct in Ldefs namespace:
// paramaters are:
//  - name of struct (in Ldefs namespace)
//  - number of bytes per  storage element
//  - 'layout' type (which determines rank
//  - name of 'padding' template (Pading or NoPadding).
//
// Normally, if the layout is chunked, you get an indirect tensor.
// Use LAYOUTDEF_CONTIG to get a contiguous tensor with a chunked layout.
//
// Do not create different configurations with the same parameters;
// all this does is generate extra duplicate code.
//
#define LAYOUTDEF(NAME, ELBYTES, LAYOUT, PAD)                                                                          \
    namespace Ldefs {                                                                                                  \
    struct API_EXPORT NAME {                                                                                           \
        using Tlayout = LAYOUT;                                                                                        \
        using storage_type = stype_for<ELBYTES>::type;                                                                 \
        static constexpr unsigned Rank = Tlayout::Rank;                                                                \
        using Pad_t = PAD<Rank>;                                                                                       \
        static constexpr bool is_chunked = Tlayout::chunk_total > 1;                                                   \
        static constexpr bool is_indirect = is_chunked;                                                                \
    };                                                                                                                 \
    }
// define a layout config which has chunked addressing, but contiguous alloc.
#define LAYOUTDEF_CONTIG(NAME, ELBYTES, LAYOUT, PAD)                                                                   \
    namespace Ldefs {                                                                                                  \
    struct API_EXPORT NAME {                                                                                           \
        using Tlayout = LAYOUT;                                                                                        \
        using storage_type = stype_for<ELBYTES>::type;                                                                 \
        static constexpr unsigned Rank = Tlayout::Rank;                                                                \
        using Pad_t = PAD<Rank>;                                                                                       \
        static constexpr bool is_chunked = Tlayout::chunk_total > 1;                                                   \
        static constexpr bool is_indirect = false;                                                                     \
    };                                                                                                                 \
    }

#define DEFINE_TYPENAMES(TYPE, NAME)                                                                                   \
    DEFINE_TYPENAME(TYPE, NAME)                                                                                        \
    DEFINE_TYPENAME_V(Vector<const TYPE *>, NAME)

// Create function that accesses the TensorTypeStruct::name that places the map of opcode ->
// typename in .rodata.
// There are two versions of this function, one below (which is called specifically for those
// tensor types which are NOT one off : RankedTensor, TensorSclrDT, LayoutTensor,
// ConcreteTensor).
// If it is one of the above four tensor types, it is declared as a static member function which
// gets created during the explicity template specialisations below.
// Behaviour:
//  - If explicity specialised and one of RankedTensor, TensorSclrDT, LayoutTensor, ConcreteTensor,
//    static member function creates the map entry in .rodata.
//  - If not explicity specialised, need to call DECLARE_TENSOR_CODE_TO_TYPENAME_STRING macro in
//    order to place entry in .rodata
template <typename T> API_FUNC_EXPORT constexpr const char *code_to_type_name()
{
    return "unknown";
}

#define DECLARE_TENSOR_CODE_TO_TYPENAME_STRING(TYPE)                                                                   \
    template <> API_FUNC_EXPORT const char *code_to_type_name<TYPE>()                                                  \
    {                                                                                                                  \
        return TensorTypeStruct<TYPE>::name;                                                                           \
    }

// macro to define a ConcreteTensor config in Tdefs namespace
//  LAYOUTNAME is a layout defined by LAYOUTDEF macro
// DTYPE and MCLASS are just dtype and memory class.
// You must use a layout with element size matching the dtype.
//
// It is possible to create different configurations with
// the same paramaters; and in this way create different
// ConcreateTensor types which behave in the same way.
//
// For instamce, QFloatCrouton and Int32Crouton have different identities
// and the same configuration.
//
#define TENSORDEF_MC(NAME, LAYOUTNAME, DTYPE, MCLASS, ENCODENAME)                                                      \
    namespace Tdefs {                                                                                                  \
    struct API_EXPORT NAME {                                                                                           \
        using Lconfig = Ldefs::LAYOUTNAME;                                                                             \
        using Tlayout = Lconfig::Tlayout;                                                                              \
        using storage_type = Lconfig::storage_type;                                                                    \
        using element_type = dtype_traits<DTYPE>::element_type;                                                        \
        static_assert(sizeof(element_type) == sizeof(storage_type), "layout has wrong element size");                  \
        using Interface_t = std::conditional_t<dtype_traits<DTYPE>::is_quant, ScaleOffsetInterface<element_type>,      \
                                               PlainInterface<element_type>>;                                          \
        static constexpr size_t Rank = Lconfig::Rank;                                                                  \
        using Pad_t = Lconfig::Pad_t;                                                                                  \
        static constexpr bool is_chunked = Lconfig::is_chunked;                                                        \
        static constexpr bool is_indirect = Lconfig::is_indirect;                                                      \
        static constexpr MemoryClass memclass = MCLASS;                                                                \
        static constexpr const char *typetag = ENCODENAME;                                                             \
    };                                                                                                                 \
    }                                                                                                                  \
    DEFINE_TYPENAMES(ConcreteTensor<Tdefs::NAME>, ENCODENAME)

#define TENSORDEF(NAME, LAYOUTNAME, DTYPE, ENCODENAME)                                                                 \
    TENSORDEF_MC(NAME, LAYOUTNAME, DTYPE, MemoryClass::Default, ENCODENAME)

// LAYOUTDEF defines a configuration
//
LAYOUTDEF(Flat_8, 1, R4FlatMemoryLayout, NoPadding)
LAYOUTDEF(Flat5D_8, 1, R5FlatMemoryLayout, NoPadding)
LAYOUTDEF(Flat_16, 2, R4FlatMemoryLayout, NoPadding)
LAYOUTDEF(Flat5D_16, 2, R5FlatMemoryLayout, NoPadding)
LAYOUTDEF(Flat_32, 4, R4FlatMemoryLayout, NoPadding)
LAYOUTDEF(Flat5D_32, 4, R5FlatMemoryLayout, NoPadding)
LAYOUTDEF(Flat6D_32, 4, R6FlatMemoryLayout, NoPadding)
LAYOUTDEF(Flat_64, 8, R4FlatMemoryLayout, NoPadding)

LAYOUTDEF(Crouton_8, 1, R4CroutonLayout, Padding)
LAYOUTDEF(Crouton_16, 2, R4Crouton2Layout, Padding)
LAYOUTDEF(Crouton_16_DeepAR4, 2, R4DeepAR4_16bLayout, Padding)
LAYOUTDEF(Crouton_16_DeepAR8, 2, R4DeepAR8_16bLayout, Padding)
LAYOUTDEF(Crouton_32, 4, R4Crouton4Layout, Padding)
LAYOUTDEF(Crouton4x1_8, 1, R4Crouton4x1Layout, Padding)
LAYOUTDEF(Crouton2x2_8, 1, R4Crouton2x2Layout, Padding)
LAYOUTDEF(WideCrouton_8, 1, R4WideCroutonLayout, Padding)
LAYOUTDEF(WideCrouton2x2_8, 1, R4WideCrouton2x2Layout, Padding)
LAYOUTDEF(WideCrouton_32, 4, R4WideCrouton4Layout, Padding)

// UNUSED LAYOUTDEF(R4Depth32_32, 4, R4Depth32MemoryLayout, NoPadding)
// UNUSED LAYOUTDEF(R4Depth32_32pad, 4, R4Depth32MemoryLayout, Padding)

LAYOUTDEF(R4Singular_8, 1, R4SingularMemoryLayout, NoPadding)
LAYOUTDEF(R4Singular_16, 2, R4SingularMemoryLayout, NoPadding)

DEFINE_TYPENAME(LayoutTensor<Ldefs::Flat_8>, "yfB")
DEFINE_TYPENAME(LayoutTensor<Ldefs::Flat5D_8>, "yf5B")
DEFINE_TYPENAME(LayoutTensor<Ldefs::Flat_16>, "yfH")
DEFINE_TYPENAME(LayoutTensor<Ldefs::Flat5D_16>, "yf5H")
DEFINE_TYPENAME(LayoutTensor<Ldefs::Flat_32>, "yfI")
DEFINE_TYPENAME(LayoutTensor<Ldefs::Flat5D_32>, "yf5I")
DEFINE_TYPENAME(LayoutTensor<Ldefs::Flat6D_32>, "yf6I")
DEFINE_TYPENAME(LayoutTensor<Ldefs::Flat_64>, "yfL")

DEFINE_TYPENAME(LayoutTensor<Ldefs::Crouton_8>, "ycB")
DEFINE_TYPENAME(LayoutTensor<Ldefs::Crouton_16>, "ycH")
DEFINE_TYPENAME(LayoutTensor<Ldefs::Crouton_32>, "ycI")
DEFINE_TYPENAME(LayoutTensor<Ldefs::Crouton_16_DeepAR4>, "ya;4H")
DEFINE_TYPENAME(LayoutTensor<Ldefs::Crouton_16_DeepAR8>, "ya;8H")

DEFINE_TYPENAME(LayoutTensor<Ldefs::R4Singular_8>, "yqB")
DEFINE_TYPENAME(LayoutTensor<Ldefs::R4Singular_16>, "yqH")

// 5D LAYOUTDEFs for Croutons
// LAYOUTDEF(Crouton_8_5D, 1, R5CroutonLayout, Padding)
// LAYOUTDEF(Crouton_16_5D, 2, R5Crouton2Layout, Padding)
// LAYOUTDEF(Crouton_32_5D, 4, R5Crouton4Layout, Padding)

// TENSORDEF
// 8-bit
TENSORDEF(QuantUint8, Flat_8, DType::QUInt8, "fB")
TENSORDEF(QuantUint8_5D, Flat5D_8, DType::QUInt8, "f5B")
TENSORDEF(QuantInt8, Flat_8, DType::QInt8, "fb")
TENSORDEF(QuantInt8_5D, Flat5D_8, DType::QInt8, "f5b")
TENSORDEF(QUint8Crouton, Crouton_8, DType::QUInt8, "cB")
TENSORDEF(QUint8Crouton4x1, Crouton4x1_8, DType::QUInt8, "c4#B")
TENSORDEF(QUint8Crouton2x2, Crouton2x2_8, DType::QUInt8, "c#B")
TENSORDEF(QUint8WideCrouton, WideCrouton_8, DType::QUInt8, "wB")
TENSORDEF(QUint8WideCrouton2x2, WideCrouton2x2_8, DType::QUInt8, "w#B")
TENSORDEF(QInt8Crouton, Crouton_8, DType::QInt8, "cb")

TENSORDEF_MC(QuantUint8_TCM, Flat_8, DType::QUInt8, MemoryClass::TCM, "FB")
TENSORDEF_MC(QuantUint8_5D_TCM, Flat5D_8, DType::QUInt8, MemoryClass::TCM, "F5B")
TENSORDEF_MC(QuantInt8_TCM, Flat_8, DType::QInt8, MemoryClass::TCM, "Fb")
TENSORDEF_MC(QuantInt8_5D_TCM, Flat5D_8, DType::QInt8, MemoryClass::TCM, "F5b")
TENSORDEF_MC(QUint8Crouton_TCM, Crouton_8, DType::QUInt8, MemoryClass::TCM, "CB")
TENSORDEF_MC(QUint8Crouton4x1_TCM, Crouton4x1_8, DType::QUInt8, MemoryClass::TCM, "C4#B")
TENSORDEF_MC(QUint8Crouton2x2_TCM, Crouton2x2_8, DType::QUInt8, MemoryClass::TCM, "C#B")
TENSORDEF_MC(QUint8WideCrouton_TCM, WideCrouton_8, DType::QUInt8, MemoryClass::TCM, "WB")
TENSORDEF_MC(QUint8WideCrouton2x2_TCM, WideCrouton2x2_8, DType::QUInt8, MemoryClass::TCM, "W#B")
TENSORDEF_MC(QInt8Crouton_TCM, Crouton_8, DType::QInt8, MemoryClass::TCM, "Cb")

// 16-bit
TENSORDEF(QuantUint16, Flat_16, DType::QUInt16, "fH")
TENSORDEF(QuantUint16_5D, Flat5D_16, DType::QUInt16, "f5H")
TENSORDEF(QuantInt16, Flat_16, DType::QInt16, "fh")
TENSORDEF(QuantInt16_5D, Flat5D_16, DType::QInt16, "f5h")
TENSORDEF(QUint16Crouton, Crouton_16, DType::QUInt16, "cH")
TENSORDEF(QUint16Crouton_AR4, Crouton_16_DeepAR4, DType::QUInt16, "a;4H")
TENSORDEF(QUint16Crouton_AR8, Crouton_16_DeepAR8, DType::QUInt16, "a;8H")
TENSORDEF(QInt16Crouton, Crouton_16, DType::QInt16, "ch")
TENSORDEF(F16Crouton, Crouton_16, DType::Float16, "ce")
TENSORDEF(F16Weights, Flat_16, DType::Float16, "fw")
TENSORDEF(PlainFloat16, Flat_16, DType::Float16, "fe")
TENSORDEF(PlainFloat16_5D, Flat5D_16, DType::Float16, "f5e")
TENSORDEF(BFloat16Crouton, Crouton_16, DType::BFloat16, "cg")
TENSORDEF(PlainBFloat16, Flat_16, DType::BFloat16, "fg")
TENSORDEF(PlainBFloat16_5D, Flat5D_16, DType::BFloat16, "f5g")

TENSORDEF_MC(QuantUint16_TCM, Flat_16, DType::QUInt16, MemoryClass::TCM, "FH")
TENSORDEF_MC(QuantUint16_5D_TCM, Flat5D_16, DType::QUInt16, MemoryClass::TCM, "F5H")
TENSORDEF_MC(QuantInt16_TCM, Flat_16, DType::QInt16, MemoryClass::TCM, "Fh")
TENSORDEF_MC(QuantInt16_5D_TCM, Flat5D_16, DType::QInt16, MemoryClass::TCM, "F5h")
TENSORDEF_MC(QUint16Crouton_TCM, Crouton_16, DType::QUInt16, MemoryClass::TCM, "CH")
TENSORDEF_MC(QUint16Crouton_AR4_TCM, Crouton_16_DeepAR4, DType::QUInt16, MemoryClass::TCM, "A;4H")
TENSORDEF_MC(QUint16Crouton_AR8_TCM, Crouton_16_DeepAR8, DType::QUInt16, MemoryClass::TCM, "A;8H")
TENSORDEF_MC(QInt16Crouton_TCM, Crouton_16, DType::QInt16, MemoryClass::TCM, "Ch")
TENSORDEF_MC(F16Crouton_TCM, Crouton_16, DType::Float16, MemoryClass::TCM, "Ce")
TENSORDEF_MC(F16Weights_TCM, Flat_16, DType::Float16, MemoryClass::TCM, "Fw")
TENSORDEF_MC(PlainFloat16_TCM, Flat_16, DType::Float16, MemoryClass::TCM, "Fe")
TENSORDEF_MC(PlainFloat16_5D_TCM, Flat5D_16, DType::Float16, MemoryClass::TCM, "F5e")
TENSORDEF_MC(BFloat16Crouton_TCM, Crouton_16, DType::BFloat16, MemoryClass::TCM, "Cg")
TENSORDEF_MC(PlainBFloat16_TCM, Flat_16, DType::BFloat16, MemoryClass::TCM, "Fg")
TENSORDEF_MC(PlainBFloat16_5D_TCM, Flat5D_16, DType::BFloat16, MemoryClass::TCM, "F5g")

// 32-bit
TENSORDEF(Int32, Flat_32, DType::Int32, "fi")
TENSORDEF(Int32_5D, Flat5D_32, DType::Int32, "f5i")
TENSORDEF(Int32_6D, Flat6D_32, DType::Int32, "f6i")
TENSORDEF(QuantInt32, Flat_32, DType::QInt32, "fs")
TENSORDEF(PlainFloat, Flat_32, DType::Float32, "ff")
TENSORDEF(PlainFloat5D, Flat5D_32, DType::Float32, "f5f")
TENSORDEF(QFloat, Flat_32, DType::Int32, "ft")
TENSORDEF(Int32Crouton, Crouton_32, DType::Int32, "ci")
TENSORDEF(QInt32Crouton, Crouton_32, DType::QInt32, "cs")
TENSORDEF(QInt32WideCrouton, WideCrouton_32, DType::QInt32, "ws")
TENSORDEF(QFloatCrouton, Crouton_32, DType::Int32, "ct")
TENSORDEF(FloatCrouton, Crouton_32, DType::Float32, "cf")

TENSORDEF_MC(Int32_TCM, Flat_32, DType::Int32, MemoryClass::TCM, "Fi")
TENSORDEF_MC(Int32_5D_TCM, Flat5D_32, DType::Int32, MemoryClass::TCM, "F5i")
TENSORDEF_MC(QInt32Crouton_TCM, Crouton_32, DType::QInt32, MemoryClass::TCM, "Cs")
TENSORDEF_MC(QInt32WideCrouton_TCM, WideCrouton_32, DType::QInt32, MemoryClass::TCM, "Ws")
TENSORDEF_MC(QuantInt32_TCM, Flat_32, DType::QInt32, MemoryClass::TCM, "Fs")
TENSORDEF_MC(PlainFloat_TCM, Flat_32, DType::Float32, MemoryClass::TCM, "Ff")
TENSORDEF_MC(PlainFloat_5D_TCM, Flat5D_32, DType::Float32, MemoryClass::TCM, "F5f")
TENSORDEF_MC(QFloat_TCM, Flat_32, DType::Int32, MemoryClass::TCM, "Ft")
TENSORDEF_MC(Int32Crouton_TCM, Crouton_32, DType::Int32, MemoryClass::TCM, "Ci")
TENSORDEF_MC(QFloatCrouton_TCM, Crouton_32, DType::Int32, MemoryClass::TCM, "Ct")
TENSORDEF_MC(FloatCrouton_TCM, Crouton_32, DType::Float32, MemoryClass::TCM, "Cf")

// 64-bit
TENSORDEF(Int64, Flat_64, DType::Int64, "fl")
TENSORDEF_MC(Int64_TCM, Flat_64, DType::Int64, MemoryClass::TCM, "Fl")

TENSORDEF(Predicate, R4Singular_8, DType::QUInt8, "qB")
TENSORDEF(Predicate16, R4Singular_16, DType::QUInt16, "qH")

DEFINE_TYPENAMES(Vector<Tensor *>, "t*")
DEFINE_TYPENAMES(TensorScalar<float>, "nf")
DEFINE_TYPENAMES(TensorScalar<NN_INT32_T>, "ni")
DEFINE_TYPENAMES(TensorScalar<NN_INT64_T>, "nl")
DEFINE_TYPENAMES(TensorShape<1>, "s1")
DEFINE_TYPENAMES(TensorShape<2>, "s2")
DEFINE_TYPENAMES(TensorShape<3>, "s3")
DEFINE_TYPENAMES(TensorShape<4>, "s4")
DEFINE_TYPENAMES(TensorShape<5>, "s5")
DEFINE_TYPENAMES(Tensor, "t")

extern template class ConcreteTensor<Tdefs::PlainFloat>;
extern template class ConcreteTensor<Tdefs::PlainFloat5D>;
extern template class ConcreteTensor<Tdefs::PlainFloat_TCM>;
extern template class ConcreteTensor<Tdefs::PlainFloat_5D_TCM>;
extern template class ConcreteTensor<Tdefs::PlainFloat16>;
extern template class ConcreteTensor<Tdefs::PlainFloat16_TCM>;
extern template class ConcreteTensor<Tdefs::PlainFloat16_5D>;
extern template class ConcreteTensor<Tdefs::PlainFloat16_5D_TCM>;
extern template class ConcreteTensor<Tdefs::PlainBFloat16>;
extern template class ConcreteTensor<Tdefs::PlainBFloat16_TCM>;
extern template class ConcreteTensor<Tdefs::PlainBFloat16_5D>;
extern template class ConcreteTensor<Tdefs::PlainBFloat16_5D_TCM>;
extern template class ConcreteTensor<Tdefs::QFloat>;
extern template class ConcreteTensor<Tdefs::QFloat_TCM>;
extern template class ConcreteTensor<Tdefs::QuantUint8>;
extern template class ConcreteTensor<Tdefs::QuantUint8_5D>;
extern template class ConcreteTensor<Tdefs::QuantInt8>;
extern template class ConcreteTensor<Tdefs::QuantInt8_5D>;
extern template class ConcreteTensor<Tdefs::QuantUint16>;
extern template class ConcreteTensor<Tdefs::QuantUint16_5D>;
extern template class ConcreteTensor<Tdefs::QuantInt16>;
extern template class ConcreteTensor<Tdefs::QuantInt16_5D>;
extern template class ConcreteTensor<Tdefs::QuantInt16_TCM>;
extern template class ConcreteTensor<Tdefs::QuantInt16_5D_TCM>;
extern template class ConcreteTensor<Tdefs::QuantInt32>;
extern template class ConcreteTensor<Tdefs::QuantInt32_TCM>;
extern template class ConcreteTensor<Tdefs::Int32>;
extern template class ConcreteTensor<Tdefs::Int32_5D>;
extern template class ConcreteTensor<Tdefs::Int32_6D>;
extern template class ConcreteTensor<Tdefs::QUint8Crouton>;
extern template class ConcreteTensor<Tdefs::QInt8Crouton>;
extern template class ConcreteTensor<Tdefs::QUint8Crouton_TCM>;
extern template class ConcreteTensor<Tdefs::QInt8Crouton_TCM>;
extern template class ConcreteTensor<Tdefs::QUint8Crouton4x1>;
extern template class ConcreteTensor<Tdefs::QUint8Crouton4x1_TCM>;
extern template class ConcreteTensor<Tdefs::QUint8Crouton2x2>;
extern template class ConcreteTensor<Tdefs::QUint8Crouton2x2_TCM>;
extern template class ConcreteTensor<Tdefs::QuantUint8_TCM>;
extern template class ConcreteTensor<Tdefs::QuantUint8_5D_TCM>;
extern template class ConcreteTensor<Tdefs::QuantUint16_TCM>;
extern template class ConcreteTensor<Tdefs::QuantUint16_5D_TCM>;
extern template class ConcreteTensor<Tdefs::QuantInt8_TCM>;
extern template class ConcreteTensor<Tdefs::QuantInt8_5D_TCM>;
extern template class ConcreteTensor<Tdefs::QInt16Crouton>;
extern template class ConcreteTensor<Tdefs::QInt16Crouton_TCM>;
extern template class ConcreteTensor<Tdefs::QUint16Crouton>;
extern template class ConcreteTensor<Tdefs::QUint16Crouton_TCM>;
extern template class ConcreteTensor<Tdefs::QUint16Crouton_AR4>;
extern template class ConcreteTensor<Tdefs::QUint16Crouton_AR4_TCM>;
extern template class ConcreteTensor<Tdefs::QUint16Crouton_AR8>;
extern template class ConcreteTensor<Tdefs::QUint16Crouton_AR8_TCM>;
extern template class ConcreteTensor<Tdefs::F16Crouton>;
extern template class ConcreteTensor<Tdefs::F16Crouton_TCM>;
extern template class ConcreteTensor<Tdefs::BFloat16Crouton>;
extern template class ConcreteTensor<Tdefs::BFloat16Crouton_TCM>;
extern template class ConcreteTensor<Tdefs::F16Weights>;
extern template class ConcreteTensor<Tdefs::F16Weights_TCM>;
extern template class ConcreteTensor<Tdefs::QInt32Crouton>;
extern template class ConcreteTensor<Tdefs::QInt32Crouton_TCM>;
extern template class ConcreteTensor<Tdefs::Int32Crouton>;
extern template class ConcreteTensor<Tdefs::Int32Crouton_TCM>;
extern template class ConcreteTensor<Tdefs::QFloatCrouton>;
extern template class ConcreteTensor<Tdefs::QFloatCrouton_TCM>;
extern template class ConcreteTensor<Tdefs::FloatCrouton>;
extern template class ConcreteTensor<Tdefs::FloatCrouton_TCM>;
extern template class ConcreteTensor<Tdefs::Int64>;
extern template class ConcreteTensor<Tdefs::Predicate>;

// standard layouts are instantiated in tensor.h
extern template class LayoutTensor<Ldefs::Flat_8>;
extern template class LayoutTensor<Ldefs::Flat_16>;
extern template class LayoutTensor<Ldefs::Flat_32>;
extern template class LayoutTensor<Ldefs::Flat5D_32>;
extern template class LayoutTensor<Ldefs::Flat6D_32>;

extern template class LayoutTensor<Ldefs::Crouton_8>;
extern template class LayoutTensor<Ldefs::Crouton_16>;
extern template class LayoutTensor<Ldefs::Crouton_32>;
extern template class LayoutTensor<Ldefs::Crouton_16_DeepAR4>;
extern template class LayoutTensor<Ldefs::Crouton_16_DeepAR8>;

// shape and scalar tensor
extern template class TensorShape<1>;
extern template class TensorShape<2>;
extern template class TensorShape<3>;
extern template class TensorShape<4>;
extern template class TensorShape<5>;

extern template class TensorSclrDT<dtype_of_type<PlainInterface<float>>()>;
extern template class TensorSclrDT<dtype_of_type<PlainInterface<NN_INT32_T>>()>;

template <typename T> // FIXME  - alias for transition
using TensorContiguous = ConcreteTensor<T>;

/////////////////////////
typedef ConcreteTensor<Tdefs::PlainFloat16_5D> PlainFloat16Tensor5D;
typedef ConcreteTensor<Tdefs::PlainFloat5D> PlainFloatTensor5D;
typedef ConcreteTensor<Tdefs::PlainFloat> PlainFloatTensor;
typedef ConcreteTensor<Tdefs::PlainFloat16> PlainFloat16Tensor;
typedef ConcreteTensor<Tdefs::QuantUint8> QuantUint8Tensor;
typedef ConcreteTensor<Tdefs::QuantUint8_5D> QuantUint8Tensor5D;
typedef ConcreteTensor<Tdefs::QuantInt8> QuantInt8Tensor;
typedef ConcreteTensor<Tdefs::QuantInt8_5D> QuantInt8Tensor5D;
typedef ConcreteTensor<Tdefs::QuantUint16> QuantUint16Tensor;
typedef ConcreteTensor<Tdefs::QuantUint16_5D> QuantUint16Tensor5D;
typedef ConcreteTensor<Tdefs::QuantInt16> QuantInt16Tensor;
typedef ConcreteTensor<Tdefs::QuantInt16_5D> QuantInt16Tensor5D;
typedef ConcreteTensor<Tdefs::QuantInt32> QuantInt32Tensor;
typedef ConcreteTensor<Tdefs::Int32> Int32Tensor;
typedef ConcreteTensor<Tdefs::Int32_5D> Int32Tensor5D;
typedef ConcreteTensor<Tdefs::Int32_6D> Int32Tensor6D;
typedef ConcreteTensor<Tdefs::Int64> Int64Tensor;
typedef ConcreteTensor<Tdefs::QUint8Crouton> QUint8CroutonTensor;
typedef ConcreteTensor<Tdefs::QInt8Crouton> QInt8CroutonTensor;
typedef ConcreteTensor<Tdefs::QUint16Crouton> QUint16CroutonTensor;
typedef ConcreteTensor<Tdefs::QUint16Crouton_AR4> QUint16CroutonTensor_AR4;
typedef ConcreteTensor<Tdefs::QUint16Crouton_AR8> QUint16CroutonTensor_AR8;
typedef ConcreteTensor<Tdefs::QInt16Crouton> QInt16CroutonTensor;
typedef ConcreteTensor<Tdefs::F16Crouton> F16CroutonTensor;
typedef ConcreteTensor<Tdefs::F16Weights> F16WeightsTensor;
typedef ConcreteTensor<Tdefs::QInt32Crouton> QInt32CroutonTensor;
typedef ConcreteTensor<Tdefs::Int32Crouton> Int32CroutonTensor;
typedef ConcreteTensor<Tdefs::QFloatCrouton> QFloatCroutonTensor;
typedef ConcreteTensor<Tdefs::QUint8WideCrouton> QUint8WideCroutonTensor;
typedef ConcreteTensor<Tdefs::QUint8WideCrouton2x2> QUint8WideCrouton2x2Tensor;
typedef ConcreteTensor<Tdefs::QInt32WideCrouton> QInt32WideCroutonTensor;
typedef ConcreteTensor<Tdefs::QUint8Crouton4x1> QUint8Crouton4x1Tensor;
typedef ConcreteTensor<Tdefs::QUint8Crouton2x2> QUint8Crouton2x2Tensor;
typedef ConcreteTensor<Tdefs::QFloat> QFloatTensor;
typedef ConcreteTensor<Tdefs::PlainBFloat16> PlainBFloat16Tensor;
typedef ConcreteTensor<Tdefs::PlainBFloat16_5D> PlainBFloat16Tensor5D;
typedef ConcreteTensor<Tdefs::BFloat16Crouton> BFloat16CroutonTensor;
typedef ConcreteTensor<Tdefs::FloatCrouton> FloatCroutonTensor;
typedef ConcreteTensor<Tdefs::Predicate> PredicateTensor;
typedef ConcreteTensor<Tdefs::Predicate16> Predicate16Tensor;

// These were once TensorContiguous
typedef ConcreteTensor<Tdefs::PlainFloat> PlainFloatContiguousTensor;
typedef ConcreteTensor<Tdefs::QFloat> QFloatContiguousTensor;

struct ModifiedDerivedTypeParent {
    using PlainFloatTensor_TCM = PlainFloatTensor;
    using PlainFloatTensor5D_TCM = PlainFloatTensor5D;
    using PlainFloat16Tensor_TCM = PlainFloat16Tensor;
    using PlainBFloat16Tensor5D_TCM = PlainBFloat16Tensor5D;
    using PlainBFloat16Tensor_TCM = PlainBFloat16Tensor;
    using PlainFloat16Tensor5D_TCM = PlainFloat16Tensor5D;
    using QFloatTensor_TCM = QFloatTensor;
    using QuantInt16Tensor_TCM = QuantInt16Tensor;
    using QuantInt16Tensor5D_TCM = QuantInt16Tensor5D;
    using QuantInt32Tensor_TCM = QuantInt32Tensor;
    using QUint8CroutonTensor_TCM = QUint8CroutonTensor;
    using QInt8CroutonTensor_TCM = QInt8CroutonTensor;
    using QUint8Crouton4x1Tensor_TCM = QUint8Crouton4x1Tensor;
    using QUint8Crouton2x2Tensor_TCM = QUint8Crouton2x2Tensor;
    using QuantUint8Tensor_TCM = QuantUint8Tensor;
    using QuantUint8Tensor5D_TCM = QuantUint8Tensor5D;
    using QuantUint16Tensor_TCM = QuantUint16Tensor;
    using QuantUint16Tensor5D_TCM = QuantUint16Tensor5D;
    using QuantInt8Tensor_TCM = QuantInt8Tensor;
    using QuantInt8Tensor5D_TCM = QuantInt8Tensor5D;
    using QUint16CroutonTensor_TCM = QUint16CroutonTensor;
    using QUint16CroutonTensor_AR4_TCM = QUint16CroutonTensor_AR4;
    using QUint16CroutonTensor_AR8_TCM = QUint16CroutonTensor_AR8;
    using QInt16CroutonTensor_TCM = QInt16CroutonTensor;
    using F16CroutonTensor_TCM = F16CroutonTensor;
    using BFloat16CroutonTensor_TCM = BFloat16CroutonTensor;
    using F16WeightsTensor_TCM = F16WeightsTensor;
    using QInt32CroutonTensor_TCM = QInt32CroutonTensor;
    using Int32CroutonTensor_TCM = Int32CroutonTensor;
    using QFloatCroutonTensor_TCM = QFloatCroutonTensor;
    using QUint8WideCroutonTensor_TCM = QUint8WideCroutonTensor;
    using QUint8WideCrouton2x2Tensor_TCM = QUint8WideCrouton2x2Tensor;
    using QInt32WideCroutonTensor_TCM = QInt32WideCroutonTensor;
    using Int32Tensor_TCM = Int32Tensor;
    using Int32Tensor5D_TCM = Int32Tensor5D;
    using Int64Tensor_TCM = Int64Tensor;
};

/////////////////////////

typedef ConcreteTensor<Tdefs::PlainFloat_TCM> PlainFloatTensor_TCM;
typedef ConcreteTensor<Tdefs::PlainFloat_5D_TCM> PlainFloatTensor5D_TCM;
typedef ConcreteTensor<Tdefs::PlainFloat16_TCM> PlainFloat16Tensor_TCM;
typedef ConcreteTensor<Tdefs::PlainFloat16_5D_TCM> PlainFloat16Tensor5D_TCM;
typedef ConcreteTensor<Tdefs::PlainBFloat16_TCM> PlainBFloat16Tensor_TCM;
typedef ConcreteTensor<Tdefs::PlainBFloat16_5D_TCM> PlainBFloat16Tensor5D_TCM;
typedef ConcreteTensor<Tdefs::QFloat_TCM> QFloatTensor_TCM;
typedef ConcreteTensor<Tdefs::QuantInt16_TCM> QuantInt16Tensor_TCM;
typedef ConcreteTensor<Tdefs::QuantInt16_5D_TCM> QuantInt16Tensor5D_TCM;
typedef ConcreteTensor<Tdefs::QuantInt32_TCM> QuantInt32Tensor_TCM;
typedef ConcreteTensor<Tdefs::QUint8Crouton_TCM> QUint8CroutonTensor_TCM;
typedef ConcreteTensor<Tdefs::QInt8Crouton_TCM> QInt8CroutonTensor_TCM;
typedef ConcreteTensor<Tdefs::QUint8Crouton4x1_TCM> QUint8Crouton4x1Tensor_TCM;
typedef ConcreteTensor<Tdefs::QUint8Crouton2x2_TCM> QUint8Crouton2x2Tensor_TCM;
typedef ConcreteTensor<Tdefs::QuantUint8_TCM> QuantUint8Tensor_TCM;
typedef ConcreteTensor<Tdefs::QuantUint8_5D_TCM> QuantUint8Tensor5D_TCM;
typedef ConcreteTensor<Tdefs::QuantUint16_TCM> QuantUint16Tensor_TCM;
typedef ConcreteTensor<Tdefs::QuantUint16_5D_TCM> QuantUint16Tensor5D_TCM;
typedef ConcreteTensor<Tdefs::QuantInt8_TCM> QuantInt8Tensor_TCM;
typedef ConcreteTensor<Tdefs::QuantInt8_5D_TCM> QuantInt8Tensor5D_TCM;
typedef ConcreteTensor<Tdefs::QUint16Crouton_TCM> QUint16CroutonTensor_TCM;
typedef ConcreteTensor<Tdefs::QUint16Crouton_AR4_TCM> QUint16CroutonTensor_AR4_TCM;
typedef ConcreteTensor<Tdefs::QUint16Crouton_AR8_TCM> QUint16CroutonTensor_AR8_TCM;
typedef ConcreteTensor<Tdefs::QInt16Crouton_TCM> QInt16CroutonTensor_TCM;
typedef ConcreteTensor<Tdefs::F16Crouton_TCM> F16CroutonTensor_TCM;
typedef ConcreteTensor<Tdefs::BFloat16Crouton_TCM> BFloat16CroutonTensor_TCM;
typedef ConcreteTensor<Tdefs::F16Weights_TCM> F16WeightsTensor_TCM;
typedef ConcreteTensor<Tdefs::QInt32Crouton_TCM> QInt32CroutonTensor_TCM;
typedef ConcreteTensor<Tdefs::Int32Crouton_TCM> Int32CroutonTensor_TCM;
typedef ConcreteTensor<Tdefs::QFloatCrouton_TCM> QFloatCroutonTensor_TCM;
typedef ConcreteTensor<Tdefs::QUint8WideCrouton_TCM> QUint8WideCroutonTensor_TCM;
typedef ConcreteTensor<Tdefs::QUint8WideCrouton2x2_TCM> QUint8WideCrouton2x2Tensor_TCM;
typedef ConcreteTensor<Tdefs::QInt32WideCrouton_TCM> QInt32WideCroutonTensor_TCM;
typedef ConcreteTensor<Tdefs::FloatCrouton_TCM> FloatCroutonTensor_TCM;

// These were once TensorContiguous
typedef ConcreteTensor<Tdefs::Int32_TCM> Int32Tensor_TCM;
typedef ConcreteTensor<Tdefs::Int32_5D_TCM> Int32Tensor5D_TCM;
typedef ConcreteTensor<Tdefs::Int64_TCM> Int64Tensor_TCM;

// typedef for layouts
typedef LayoutTensor<Ldefs::Flat_8> LayoutFlat_8;
typedef LayoutTensor<Ldefs::Flat5D_8> LayoutFlat5D_8;
typedef LayoutTensor<Ldefs::Flat_16> LayoutFlat_16;
typedef LayoutTensor<Ldefs::Flat5D_16> LayoutFlat5D_16;
typedef LayoutTensor<Ldefs::Flat_32> LayoutFlat_32;
typedef LayoutTensor<Ldefs::Flat5D_32> LayoutFlat5D_32;
typedef LayoutTensor<Ldefs::Flat_64> LayoutFlat_64;

// 'standard' crouton layouts.
typedef LayoutTensor<Ldefs::Crouton_8> LayoutCrouton_8; // [1,8,8,32]
typedef LayoutTensor<Ldefs::WideCrouton_8> LayoutWideCrouton_8; // [1,2,32,32]
typedef LayoutTensor<Ldefs::Crouton_16> LayoutCrouton_16; // [1,8,4,32] interleaved
typedef LayoutTensor<Ldefs::Crouton_16_DeepAR4> LayoutCrouton_16_DeepAR4; // [1,1,4,256] interleaved
typedef LayoutTensor<Ldefs::Crouton_16_DeepAR8> LayoutCrouton_16_DeepAR8; // [1,1,8,128] interleaved
typedef LayoutTensor<Ldefs::Crouton_32> LayoutCrouton_32; // [1,8,2,32]
typedef LayoutTensor<Ldefs::WideCrouton_32> LayoutWideCrouton_32; // [1,2,8,32]

typedef LayoutTensor<Ldefs::Crouton4x1_8> LayoutCrouton4x1_8;
typedef LayoutTensor<Ldefs::Crouton2x2_8> LayoutCrouton2x2_8;
typedef LayoutTensor<Ldefs::WideCrouton2x2_8> LayoutWideCrouton2x2_8;

using TypicalTensors =
        std::tuple<PlainFloatTensor, PlainFloatTensor5D, PlainFloat16Tensor, QuantUint8Tensor, QuantUint8Tensor5D,
                   QuantInt8Tensor, QuantInt8Tensor5D, QuantUint16Tensor, QuantUint16Tensor5D, QuantInt16Tensor,
                   QuantInt32Tensor, Int32Tensor, Int32Tensor5D, Int32Tensor6D, QUint8CroutonTensor, QInt8CroutonTensor,
                   QUint8Crouton4x1Tensor, QUint8Crouton2x2Tensor, QUint16CroutonTensor, QInt16CroutonTensor,
                   QUint16CroutonTensor_AR4, QUint16CroutonTensor_AR8, QInt32CroutonTensor, QFloatTensor,
                   QFloatCroutonTensor, Int32CroutonTensor, PlainFloat16Tensor_TCM, PlainFloat16Tensor5D, Int64Tensor,
                   QuantInt16Tensor5D, PlainBFloat16Tensor, PlainBFloat16Tensor5D, BFloat16CroutonTensor>;

namespace hnnx {
// these tensor types are 'pre-registered' for deserialize
// clang-format off
using CoreTensors =
        std::tuple<PlainFloatTensor, PlainFloatTensor5D, PlainFloat16Tensor, Int32Tensor, Int32Tensor5D, Int32Tensor6D,
                   PlainFloatTensor_TCM, PlainFloatTensor5D_TCM, Int32Tensor_TCM, QuantUint8Tensor, QuantUint8Tensor5D,
                   QuantInt8Tensor, QuantInt8Tensor5D, QuantUint8Tensor_TCM, QuantUint8Tensor5D_TCM,
                   QuantInt8Tensor_TCM, QuantInt8Tensor5D_TCM, QuantUint16Tensor, QuantUint16Tensor5D, QuantInt16Tensor,
                   QuantUint16Tensor_TCM, QuantUint16Tensor5D_TCM, QuantInt16Tensor_TCM, QuantInt32Tensor,
                   QUint8CroutonTensor, QuantInt32Tensor_TCM, QUint8CroutonTensor_TCM, QInt8CroutonTensor,
                   QUint16CroutonTensor, QInt8CroutonTensor_TCM, QUint16CroutonTensor_TCM, QInt32CroutonTensor,
                   QUint16CroutonTensor_AR4_TCM, QUint16CroutonTensor_AR8_TCM,
                   QUint16CroutonTensor_AR4, QUint16CroutonTensor_AR8,
                   QInt16CroutonTensor, QInt16CroutonTensor_TCM, QInt32CroutonTensor_TCM, QInt32WideCroutonTensor,
                   QInt32WideCroutonTensor_TCM, QFloatTensor, QFloatCroutonTensor, Int32CroutonTensor,
                   Int32CroutonTensor_TCM,
                   F16CroutonTensor, F16CroutonTensor_TCM,
                   QUint8WideCroutonTensor, QUint8WideCroutonTensor_TCM, QUint8Crouton2x2Tensor_TCM,
                   QUint8WideCrouton2x2Tensor_TCM, PlainFloat16Tensor_TCM, PlainFloat16Tensor5D, Int32Tensor5D_TCM,
                   Int64Tensor, Int64Tensor_TCM,
                   QuantInt16Tensor5D_TCM, QuantInt16Tensor5D, PlainBFloat16Tensor, PlainBFloat16Tensor_TCM, PlainBFloat16Tensor5D, PlainBFloat16Tensor5D_TCM, 
                   BFloat16CroutonTensor, BFloat16CroutonTensor_TCM>;
// clang-format on

} // namespace hnnx

POP_VISIBILITY()

#endif // HEXNN_TENSOR_DEFINITIONS_H
