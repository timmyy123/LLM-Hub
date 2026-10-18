// ==============================================================================
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear
//
// ==============================================================================

/**
 * @file defopt2qhpi.h
 * @brief Helper library for migrating legacy DEF_OPT / DEF_PACKAGE_OP code to QHPI.
 *
 * This header-only library provides inline C++ convenience functions built on
 * the public QHPI C API (qhpi.h). It is intended for use by developers
 * converting legacy operator specifications (DEF_OPT rules, DEF_PACKAGE_OP
 * registrations, DEF_TENSOR_PROPERTIES, tiling rules) to the QHPI interface.
 *
 * All functions live in namespace @c qhpi and depend only on qhpi.h.
 *
 * @see docs/qhpi/defopt2qhpi.md for migration guide and worked examples.
 */

#ifndef DEFOPT2QHPI_H
#define DEFOPT2QHPI_H

#include "qhpi.h"

#include <array>
#include <cassert>
#include <cstring>
#include <initializer_list>
#include <optional>
#include <vector>

namespace qhpi {

// ============================================================================
// Core wrapper type
// ============================================================================

/**
 * @brief Convenience wrapper around QHPI_OpRef.
 *
 * Can be constructed from a QHPI_OpRef (preserving output_number) or from a
 * QHPI_Op pointer (defaulting to output 0, asserting single output).
 */
struct OpOrRef : QHPI_OpRef {
    OpOrRef(const QHPI_OpRef r) : QHPI_OpRef{r.op, r.output_number} {}
    OpOrRef(const QHPI_Op *p) : QHPI_OpRef{p, 0} { assert(qhpi_op_num_outputs(p) == 1); }
    OpOrRef() : QHPI_OpRef{nullptr, 0} {}
    OpOrRef(const OpOrRef &) = default;
};

// ============================================================================
// Type checking  (replaces IS_QUINT8, IS_FLOAT16, etc.)
// ============================================================================

/** Element type of the first output. */
inline QHPI_Element_Type output_type(const QHPI_Op *op)
{
    return qhpi_op_output(op, 0).type;
}

/** Element type of output at @p idx. */
inline QHPI_Element_Type output_type(const QHPI_Op *op, uint32_t idx)
{
    return qhpi_op_output(op, idx).type;
}

/** Element type of input at @p idx. */
inline QHPI_Element_Type input_type(const QHPI_Op *op, uint32_t idx)
{
    QHPI_OpRef in = qhpi_op_input(op, idx);
    return qhpi_op_output(in.op, in.output_number).type;
}

inline bool is_quint8(const QHPI_Op *op)
{
    return output_type(op) == QHPI_QUINT8;
}
inline bool is_quint16(const QHPI_Op *op)
{
    return output_type(op) == QHPI_QUINT16;
}
inline bool is_qint8(const QHPI_Op *op)
{
    return output_type(op) == QHPI_QINT8;
}
inline bool is_qint16(const QHPI_Op *op)
{
    return output_type(op) == QHPI_QINT16;
}
inline bool is_float32(const QHPI_Op *op)
{
    return output_type(op) == QHPI_FLOAT32;
}
inline bool is_float16(const QHPI_Op *op)
{
    return output_type(op) == QHPI_FLOAT16;
}
inline bool is_int32(const QHPI_Op *op)
{
    return output_type(op) == QHPI_INT32;
}

inline bool is_quint8_input(const QHPI_Op *op, uint32_t idx)
{
    return input_type(op, idx) == QHPI_QUINT8;
}
inline bool is_quint16_input(const QHPI_Op *op, uint32_t idx)
{
    return input_type(op, idx) == QHPI_QUINT16;
}
inline bool is_float32_input(const QHPI_Op *op, uint32_t idx)
{
    return input_type(op, idx) == QHPI_FLOAT32;
}
inline bool is_float16_input(const QHPI_Op *op, uint32_t idx)
{
    return input_type(op, idx) == QHPI_FLOAT16;
}
inline bool is_qint8_input(const QHPI_Op *op, uint32_t idx)
{
    return input_type(op, idx) == QHPI_QINT8;
}
inline bool is_qint16_input(const QHPI_Op *op, uint32_t idx)
{
    return input_type(op, idx) == QHPI_QINT16;
}
inline bool is_int32_input(const QHPI_Op *op, uint32_t idx)
{
    return input_type(op, idx) == QHPI_INT32;
}

/** True when all inputs and the first output share the same element type. */
inline bool all_same_type(const QHPI_Op *op)
{
    const QHPI_Element_Type t = output_type(op);
    const uint32_t n = qhpi_op_num_inputs(op);
    for (uint32_t i = 0; i < n; ++i)
        if (input_type(op, i) != t) return false;
    return true;
}

// ============================================================================
// Output / shape accessors
// ============================================================================

/** First output definition (most common case). */
inline QHPI_OutputDef output(const QHPI_Op *op)
{
    return qhpi_op_output(op, 0);
}

/** Output definition for an OpOrRef. */
inline QHPI_OutputDef output(OpOrRef ref)
{
    return qhpi_op_output(ref.op, ref.output_number);
}

/** Output definition at arbitrary index. */
inline QHPI_OutputDef output(const QHPI_Op *op, uint32_t idx)
{
    return qhpi_op_output(op, idx);
}

// Dimension accessors for rank-4 tensors (batch=0, height=1, width=2, depth=3)
inline uint32_t batch(const QHPI_Shape &s)
{
    return s.dims[0];
}
inline uint32_t height(const QHPI_Shape &s)
{
    return s.dims[1];
}
inline uint32_t width(const QHPI_Shape &s)
{
    return s.dims[2];
}
inline uint32_t depth(const QHPI_Shape &s)
{
    return s.dims[3];
}

inline uint32_t &batch(QHPI_Shape &s)
{
    return s.dims[0];
}
inline uint32_t &height(QHPI_Shape &s)
{
    return s.dims[1];
}
inline uint32_t &width(QHPI_Shape &s)
{
    return s.dims[2];
}
inline uint32_t &depth(QHPI_Shape &s)
{
    return s.dims[3];
}

/** Total number of elements in a shape. */
inline uint32_t total_elements(const QHPI_Shape &s)
{
    uint32_t total = 1;
    for (uint32_t i = 0; i < s.rank; ++i)
        total *= s.dims[i];
    return total;
}

// ============================================================================
// Shape construction  (replaces gen_Shape, gen_ShapeOf)
// ============================================================================

/** Create a rank-4 shape. Replaces gen_Shape(b,h,w,d). */
inline QHPI_Shape make_shape(uint32_t b, uint32_t h, uint32_t w, uint32_t d)
{
    QHPI_Shape s;
    s.rank = 4;
    s.dims[0] = b;
    s.dims[1] = h;
    s.dims[2] = w;
    s.dims[3] = d;
    return s;
}

/** Copy shape from an op's first output. Replaces gen_ShapeOf("operand"). */
inline QHPI_Shape shape_of(const QHPI_Op *op)
{
    return qhpi_op_output(op, 0).shape;
}

/** Copy shape from an OpOrRef. */
inline QHPI_Shape shape_of(OpOrRef ref)
{
    return qhpi_op_output(ref.op, ref.output_number).shape;
}

/** Return a copy of @p def with its shape replaced. */
inline QHPI_OutputDef with_shape(QHPI_OutputDef def, const QHPI_Shape &shape)
{
    def.shape = shape;
    return def;
}

/** Return a copy of @p def with a rank-4 shape. */
inline QHPI_OutputDef with_shape(QHPI_OutputDef def, uint32_t b, uint32_t h, uint32_t w, uint32_t d)
{
    def.shape = make_shape(b, h, w, d);
    return def;
}

// ============================================================================
// Input / output accessors  (structured-binding friendly)
// ============================================================================

/**
 * Get the first N inputs as a std::array.
 *
 * @pre N <= qhpi_op_num_inputs(op)
 * @note If N exceeds the actual input count, the trailing slots of the
 *       returned array are zero-initialized (null QHPI_OpRef). This function
 *       does not fail on out-of-range N. When N is not statically known to
 *       match the op's arity, guard the call with qhpi_op_num_inputs() or
 *       qhpi::op_is(op, name, N) first.
 */
template <unsigned N> inline std::array<QHPI_OpRef, N> get_inputs(const QHPI_Op *op)
{
    std::array<QHPI_OpRef, N> inputs{};
    const unsigned m = (N < qhpi_op_num_inputs(op)) ? N : qhpi_op_num_inputs(op);
    for (unsigned i = 0; i < m; ++i)
        inputs[i] = qhpi_op_input(op, i);
    return inputs;
}

/**
 * Get the first N output definitions as a std::array.
 *
 * @pre N <= qhpi_op_num_outputs(op)
 * @note If N exceeds the actual output count, the trailing slots of the
 *       returned array are value-initialized (zeroed QHPI_OutputDef). This
 *       function does not fail on out-of-range N. Guard with
 *       qhpi_op_num_outputs() when N is not statically known.
 */
template <unsigned N> inline std::array<QHPI_OutputDef, N> get_outputs(const QHPI_Op *op)
{
    std::array<QHPI_OutputDef, N> outputs{};
    const unsigned m = (N < qhpi_op_num_outputs(op)) ? N : qhpi_op_num_outputs(op);
    for (unsigned i = 0; i < m; ++i)
        outputs[i] = qhpi_op_output(op, i);
    return outputs;
}

// ============================================================================
// Constant generation  (replaces gen_ConstScalar_*, gen_ConstArr_*)
// ============================================================================

/** Create a scalar float32 constant. Replaces gen_ConstScalar_f32(value). */
inline QHPI_OpRef make_const_scalar_f32(const QHPI_Op *root, float value)
{
    QHPI_OutputDef out = {QHPI_FLOAT32, {0, 1.0f}, make_shape(1, 1, 1, 1)};
    return qhpi_op_reference(qhpi_op_create_constant(root, &out, sizeof(float), &value), 0);
}

/** Create a scalar int32 constant. Replaces gen_ConstScalar_i32(value). */
inline QHPI_OpRef make_const_scalar_i32(const QHPI_Op *root, int32_t value)
{
    QHPI_OutputDef out = {QHPI_INT32, {0, 0.0f}, {0, {1, 1, 1, 1}}};
    return qhpi_op_reference(qhpi_op_create_constant(root, &out, sizeof(int32_t), &value), 0);
}

/** Create a 1-D float32 constant filled with @p value. Replaces gen_ConstArr_f32. */
inline QHPI_OpRef make_const_1d_f32(const QHPI_Op *root, float value, uint32_t size)
{
    std::vector<float> fill(size, value);
    QHPI_OutputDef out = {QHPI_FLOAT32, {0, 1.0f}, make_shape(1, 1, 1, size)};
    return qhpi_op_reference(qhpi_op_create_constant(root, &out, sizeof(float) * size, fill.data()), 0);
}

/** Create a 1-D int32 constant filled with @p value. Replaces gen_ConstArr_i32. */
inline QHPI_OpRef make_const_1d_i32(const QHPI_Op *root, int32_t value, uint32_t size)
{
    std::vector<int32_t> fill(size, value);
    QHPI_OutputDef out = {QHPI_INT32, {0, 0.0f}, make_shape(1, 1, 1, size)};
    return qhpi_op_reference(qhpi_op_create_constant(root, &out, sizeof(int32_t) * size, fill.data()), 0);
}

/** Create a shape-only op (no data). */
inline QHPI_OpRef make_shape_op(const QHPI_Op *root, uint32_t b, uint32_t h, uint32_t w, uint32_t d)
{
    QHPI_Shape s = make_shape(b, h, w, d);
    return qhpi_op_create_shape(root, &s);
}

// ============================================================================
// Op creation helpers
// ============================================================================

/** Create an op from a std::array of inputs and a single output. */
template <size_t N>
inline const QHPI_Op *create(const QHPI_Op *root, const char *name, const std::array<QHPI_OpRef, N> &inputs,
                             const QHPI_OutputDef &out)
{
    return qhpi_op_create(root, name, N, inputs.data(), 1, &out);
}

/** Create an op from a C array of inputs and a single output. */
template <size_t N>
inline const QHPI_Op *create(const QHPI_Op *root, const char *name, const QHPI_OpRef (&inputs)[N],
                             const QHPI_OutputDef &out)
{
    return qhpi_op_create(root, name, N, inputs, 1, &out);
}

/** Create an op from a std::array of inputs and multiple outputs. */
template <size_t N, size_t M>
inline const QHPI_Op *create(const QHPI_Op *root, const char *name, const std::array<QHPI_OpRef, N> &inputs,
                             const std::array<QHPI_OutputDef, M> &outputs)
{
    return qhpi_op_create(root, name, N, inputs.data(), M, outputs.data());
}

/** Create an op from an initializer_list of OpOrRef and a single output. */
inline const QHPI_Op *create(const QHPI_Op *root, const char *name, std::initializer_list<OpOrRef> inputs,
                             const QHPI_OutputDef &out)
{
    // Small-buffer optimization: stack-allocate up to 8 inputs.
    QHPI_OpRef buf[8];
    std::vector<QHPI_OpRef> heap;
    QHPI_OpRef *ptr = buf;
    const size_t n = inputs.size();
    if (n > 8) {
        heap.assign(inputs.begin(), inputs.end());
        ptr = heap.data();
    } else {
        size_t i = 0;
        for (const auto &r : inputs)
            buf[i++] = r;
    }
    return qhpi_op_create(root, name, static_cast<uint32_t>(n), ptr, 1, &out);
}

/** Create an op and return a reference to its first output. */
template <size_t N>
inline QHPI_OpRef create_ref(const QHPI_Op *root, const char *name, const std::array<QHPI_OpRef, N> &inputs,
                             const QHPI_OutputDef &out)
{
    return qhpi_op_reference(create(root, name, inputs, out), 0);
}

/** Create an op from initializer_list and return a reference to its first output. */
inline QHPI_OpRef create_ref(const QHPI_Op *root, const char *name, std::initializer_list<OpOrRef> inputs,
                             const QHPI_OutputDef &out)
{
    return qhpi_op_reference(create(root, name, inputs, out), 0);
}

// ============================================================================
// Common rewrite patterns
// ============================================================================

/**
 * Rename an op: create a new op with @p new_name, same inputs and output.
 * Replaces: DEF_OPT(TRANSLATE+N, Op(QNN_X, "in"), OK, Op("Y", "in"))
 */
template <unsigned N> inline const QHPI_Op *rename(const QHPI_Op *op, const char *new_name)
{
    assert(qhpi_op_num_inputs(op) == N);
    assert(qhpi_op_num_outputs(op) == 1);
    return create(op, new_name, get_inputs<N>(op), qhpi_op_output(op, 0));
}

/**
 * Rename an op: create a new op with @p new_name, same inputs and output.
 * Runtime-arity convenience for pure-rename rewrites where the input count
 * is whatever the source op has. Preserves all inputs in order and the
 * first output's definition unchanged.
 */
inline const QHPI_Op *rename(const QHPI_Op *op, const char *new_name)
{
    assert(qhpi_op_num_outputs(op) == 1);
    const uint32_t n = qhpi_op_num_inputs(op);
    std::vector<QHPI_OpRef> inputs(n);
    for (uint32_t i = 0; i < n; ++i)
        inputs[i] = qhpi_op_input(op, i);
    const QHPI_OutputDef out = qhpi_op_output(op, 0);
    return qhpi_op_create(op, new_name, n, inputs.data(), 1, &out);
}

// ============================================================================
// FP16 conversion  (replaces MAKE_OP_FP16_AND_INSERT_CAST + relaxed_precision_flag)
// ============================================================================

/** Insert a Cast node converting @p in to @p target_type. */
inline const QHPI_Op *make_cast(const QHPI_Op *root, OpOrRef in, QHPI_Element_Type target_type)
{
    QHPI_OutputDef out = qhpi::output(in);
    out.type = target_type;
    QHPI_OpRef ref = in;
    return qhpi_op_create(root, "q::Cast", 1, &ref, 1, &out);
}

/**
 * Convert a single-input, single-output op to FP16 with surrounding casts.
 * Replaces MAKE_OP_FP16_AND_INSERT_CAST(Op(QNN_X, CAST_TO_FP16("In"))).
 */
inline const QHPI_Op *make_op_fp16_and_insert_cast(const QHPI_Op *op)
{
    assert(qhpi_op_num_inputs(op) == 1);
    assert(qhpi_op_num_outputs(op) == 1);
    // Cast input to FP16
    const QHPI_Op *cast_in = make_cast(op, qhpi_op_input(op, 0), QHPI_FLOAT16);
    // Recreate op with FP16 output
    QHPI_OutputDef fp16_out = qhpi_op_output(op, 0);
    fp16_out.type = QHPI_FLOAT16;
    QHPI_OpRef fp16_ref = {cast_in, 0};
    const QHPI_Op *fp16_op = qhpi_op_create(op, qhpi_op_name(op), 1, &fp16_ref, 1, &fp16_out);
    // Cast output back to FP32
    return make_cast(op, OpOrRef{fp16_op}, QHPI_FLOAT32);
}

/**
 * Check relaxed_precision_flag and convert to FP16 if applicable.
 * Returns the converted op, or nullptr if no conversion was done.
 */
inline const QHPI_Op *convert_to_fp16(const QHPI_Op *op)
{
    QHPI_OutputDef out = qhpi_op_output(op, 0);
    int32_t flag = 0;
    if (out.type == QHPI_FLOAT32 && qhpi_option_int(op, "relaxed_precision_flag", &flag) && flag)
        return make_op_fp16_and_insert_cast(op);
    return nullptr;
}

// ============================================================================
// Tiling helpers  (replaces DEF_AUTOSPLITIM + TYPICAL_SLICE)
// ============================================================================

/**
 * Slice an input tensor. Equivalent to TYPICAL_SLICE("input", "I").
 * This is a thin wrapper for naming clarity in migration context.
 */
inline QHPI_OpRef typical_slice(QHPI_OpRef input, const QHPI_Shape *start, const QHPI_Shape *extent)
{
    return qhpi_op_slice(input, start, extent);
}

/**
 * Build a tile for a simple N-input, single-output elementwise op.
 * Each input is sliced with the same start/extent, and the output shape
 * is set to *extent.
 *
 * Replaces: DEF_AUTOSPLITIM(... Op("Name", TYPICAL_SLICE("in0","I"), TYPICAL_SLICE("in1","I"), ...))
 */
template <unsigned N>
inline const QHPI_Op *build_typical_tile(const QHPI_Op *op, const QHPI_Shape *start, const QHPI_Shape *extent)
{
    assert(qhpi_op_num_inputs(op) == N);
    assert(qhpi_op_num_outputs(op) == 1);
    std::array<QHPI_OpRef, N> sliced;
    for (unsigned i = 0; i < N; ++i)
        sliced[i] = qhpi_op_slice(qhpi_op_input(op, i), start, extent);
    QHPI_OutputDef tile_out = qhpi_op_output(op, 0);
    tile_out.shape = *extent;
    return qhpi_op_create(op, qhpi_op_name(op), N, sliced.data(), 1, &tile_out);
}

/**
 * Build the QHPI_Shape that a `shape_required` callback returns.
 * Use 0 for "no constraint" on a dimension, QHPI_NO_TILING to prevent tiling
 * on that dimension. The op association is established by the
 * QHPI_OpInfo_v1 registration that names this callback, not by the shape
 * value itself. Mirrors the legacy TILE_SHAPE_REQUIRED form that attached a
 * shape to an op name via the DEF_OPT registry.
 */
inline QHPI_Shape make_shape_required(uint32_t b, uint32_t h, uint32_t w, uint32_t d)
{
    return make_shape(b, h, w, d);
}

// ============================================================================
// Signature construction  (replaces DEF_TENSOR_PROPERTIES + type dispatch)
// ============================================================================

/** Build a QHPI_Tensor_Signature_v1 with specified attributes. */
inline constexpr QHPI_Tensor_Signature_v1 make_signature(uint32_t type, uint32_t layout, uint8_t storage,
                                                         uint8_t placement)
{
    return {type, layout, storage, placement};
}

// Pre-built signature constants for common tensor type mappings.
// These replace the implicit type dispatch from DEF_PACKAGE_OP template arguments.

// Flat, DDR only
static constexpr QHPI_Tensor_Signature_v1 SIG_FLAT4_QUINT8_DDR = {QHPI_QUINT8, QHPI_LAYOUT_FLAT_4, QHPI_STORAGE_DIRECT,
                                                                  QHPI_MEM_LOC_DDR_ONLY};
static constexpr QHPI_Tensor_Signature_v1 SIG_FLAT4_QUINT16_DDR = {QHPI_QUINT16, QHPI_LAYOUT_FLAT_4,
                                                                   QHPI_STORAGE_DIRECT, QHPI_MEM_LOC_DDR_ONLY};
static constexpr QHPI_Tensor_Signature_v1 SIG_FLAT4_INT32_DDR = {QHPI_INT32, QHPI_LAYOUT_FLAT_4, QHPI_STORAGE_DIRECT,
                                                                 QHPI_MEM_LOC_DDR_ONLY};
static constexpr QHPI_Tensor_Signature_v1 SIG_FLAT4_FLOAT32_DDR = {QHPI_FLOAT32, QHPI_LAYOUT_FLAT_4,
                                                                   QHPI_STORAGE_DIRECT, QHPI_MEM_LOC_DDR_ONLY};
static constexpr QHPI_Tensor_Signature_v1 SIG_FLAT4_FLOAT16_DDR = {QHPI_FLOAT16, QHPI_LAYOUT_FLAT_4,
                                                                   QHPI_STORAGE_DIRECT, QHPI_MEM_LOC_DDR_ONLY};

// Flat, TCM only
static constexpr QHPI_Tensor_Signature_v1 SIG_FLAT4_QUINT8_TCM = {QHPI_QUINT8, QHPI_LAYOUT_FLAT_4, QHPI_STORAGE_DIRECT,
                                                                  QHPI_MEM_LOC_TCM_ONLY};
static constexpr QHPI_Tensor_Signature_v1 SIG_FLAT4_QUINT16_TCM = {QHPI_QUINT16, QHPI_LAYOUT_FLAT_4,
                                                                   QHPI_STORAGE_DIRECT, QHPI_MEM_LOC_TCM_ONLY};
static constexpr QHPI_Tensor_Signature_v1 SIG_FLAT4_INT32_TCM = {QHPI_INT32, QHPI_LAYOUT_FLAT_4, QHPI_STORAGE_DIRECT,
                                                                 QHPI_MEM_LOC_TCM_ONLY};
static constexpr QHPI_Tensor_Signature_v1 SIG_FLAT4_FLOAT16_TCM = {QHPI_FLOAT16, QHPI_LAYOUT_FLAT_4,
                                                                   QHPI_STORAGE_DIRECT, QHPI_MEM_LOC_TCM_ONLY};

// Flat, DDR or TCM (replaces WhenFitsIfTCM pattern)
static constexpr QHPI_Tensor_Signature_v1 SIG_FLAT4_QUINT8_DDR_OR_TCM = {QHPI_QUINT8, QHPI_LAYOUT_FLAT_4,
                                                                         QHPI_STORAGE_DIRECT, QHPI_MEM_LOC_DDR_OR_TCM};
static constexpr QHPI_Tensor_Signature_v1 SIG_FLAT4_QUINT16_DDR_OR_TCM = {QHPI_QUINT16, QHPI_LAYOUT_FLAT_4,
                                                                          QHPI_STORAGE_DIRECT, QHPI_MEM_LOC_DDR_OR_TCM};
static constexpr QHPI_Tensor_Signature_v1 SIG_FLAT4_INT32_DDR_OR_TCM = {QHPI_INT32, QHPI_LAYOUT_FLAT_4,
                                                                        QHPI_STORAGE_DIRECT, QHPI_MEM_LOC_DDR_OR_TCM};
static constexpr QHPI_Tensor_Signature_v1 SIG_FLAT4_FLOAT16_DDR_OR_TCM = {QHPI_FLOAT16, QHPI_LAYOUT_FLAT_4,
                                                                          QHPI_STORAGE_DIRECT, QHPI_MEM_LOC_DDR_OR_TCM};

// Crouton, DDR only
static constexpr QHPI_Tensor_Signature_v1 SIG_CROUTON8_QUINT8_DDR = {QHPI_QUINT8, QHPI_LAYOUT_CROUTON_8,
                                                                     QHPI_STORAGE_INDIRECT, QHPI_MEM_LOC_DDR_ONLY};
static constexpr QHPI_Tensor_Signature_v1 SIG_CROUTON16_QUINT16_DDR = {QHPI_QUINT16, QHPI_LAYOUT_CROUTON_16,
                                                                       QHPI_STORAGE_INDIRECT, QHPI_MEM_LOC_DDR_ONLY};
static constexpr QHPI_Tensor_Signature_v1 SIG_CROUTON16_FLOAT16_DDR = {QHPI_FLOAT16, QHPI_LAYOUT_CROUTON_16,
                                                                       QHPI_STORAGE_INDIRECT, QHPI_MEM_LOC_DDR_ONLY};

// Crouton, TCM only
static constexpr QHPI_Tensor_Signature_v1 SIG_CROUTON8_QUINT8_TCM = {QHPI_QUINT8, QHPI_LAYOUT_CROUTON_8,
                                                                     QHPI_STORAGE_INDIRECT, QHPI_MEM_LOC_TCM_ONLY};
static constexpr QHPI_Tensor_Signature_v1 SIG_CROUTON16_FLOAT16_TCM = {QHPI_FLOAT16, QHPI_LAYOUT_CROUTON_16,
                                                                       QHPI_STORAGE_INDIRECT, QHPI_MEM_LOC_TCM_ONLY};

// Crouton, DDR or TCM
static constexpr QHPI_Tensor_Signature_v1 SIG_CROUTON8_QUINT8_DDR_OR_TCM = {
        QHPI_QUINT8, QHPI_LAYOUT_CROUTON_8, QHPI_STORAGE_INDIRECT, QHPI_MEM_LOC_DDR_OR_TCM};
static constexpr QHPI_Tensor_Signature_v1 SIG_CROUTON16_FLOAT16_DDR_OR_TCM = {
        QHPI_FLOAT16, QHPI_LAYOUT_CROUTON_16, QHPI_STORAGE_INDIRECT, QHPI_MEM_LOC_DDR_OR_TCM};

// Wildcard (any type/layout) — for variable-type inputs
static constexpr QHPI_Tensor_Signature_v1 SIG_ANY = {QHPI_ELEMENT_TYPE_ANY, QHPI_LAYOUT_ANY,
                                                     QHPI_STORAGE_DIRECT_OR_INDIRECT, QHPI_MEM_LOC_DDR_OR_TCM};

// ============================================================================
// Option access  (replaces OPTION_INT, OPTION_BOOL)
// ============================================================================

/** Read an integer option. Returns @p default_val if the option is not set. */
inline int32_t option_int(const QHPI_Op *op, const char *name, int32_t default_val = 0)
{
    int32_t val = default_val;
    qhpi_option_int(op, name, &val);
    return val;
}

/** Read a boolean option (true if option_int != 0). */
inline bool option_bool(const QHPI_Op *op, const char *name)
{
    return option_int(op, name) != 0;
}

/**
 * True when integer option @p name is set and equals @p expected.
 *
 * Returns false when the option is not set on the op, even if @p expected
 * happens to equal the helper's default. Replaces the legacy
 * GE(OPTION_INT("name"), value) / EQ(OPTION_INT("name"), value) predicates.
 */
inline bool check_option(const QHPI_Op *op, const char *name, int32_t expected)
{
    int32_t val = 0;
    if (qhpi_option_int(op, name, &val) != QHPI_SUCCESS) return false;
    return val == expected;
}

// ============================================================================
// Quantization parameter access  (replaces STEPSIZE_OF, ZERO_OFFSET_OF)
// ============================================================================

/** Get quantization parameters of the first output. */
inline QHPI_Quant_Parameters quant_of(const QHPI_Op *op)
{
    return qhpi_op_output(op, 0).quant_parameters;
}

/** Get quantization parameters of input at @p idx. */
inline QHPI_Quant_Parameters quant_of_input(const QHPI_Op *op, uint32_t idx)
{
    QHPI_OpRef in = qhpi_op_input(op, idx);
    return qhpi_op_output(in.op, in.output_number).quant_parameters;
}

/** Stepsize of the first output. Replaces STEPSIZE_OF("*"). */
inline float stepsize_of(const QHPI_Op *op)
{
    return quant_of(op).stepsize;
}

/** Zero offset of the first output. Replaces ZERO_OFFSET_OF("*"). */
inline int32_t zero_offset_of(const QHPI_Op *op)
{
    return quant_of(op).zero_offset;
}

/** Stepsize of input at @p idx. Replaces STEPSIZE_OF("input_name"). */
inline float stepsize_of_input(const QHPI_Op *op, uint32_t idx)
{
    return quant_of_input(op, idx).stepsize;
}

/** Zero offset of input at @p idx. Replaces ZERO_OFFSET_OF("input_name"). */
inline int32_t zero_offset_of_input(const QHPI_Op *op, uint32_t idx)
{
    return quant_of_input(op, idx).zero_offset;
}

// ============================================================================
// Graph query helpers  (replaces PRODUCER_FOR, has_only_one_consumer, etc.)
// ============================================================================

/** True if op has exactly one consumer. Replaces has_only_one_consumer(op). */
inline bool has_one_consumer(const QHPI_Op *op)
{
    return qhpi_op_num_uses(op) == 1;
}

/** True if op is a constant. Replaces OPCONST("X") check. */
inline bool is_constant(const QHPI_Op *op)
{
    return qhpi_op_is_constant(op);
}

/** True if input at @p idx is a constant. */
inline bool is_constant_input(const QHPI_Op *op, uint32_t idx)
{
    return qhpi_op_is_constant(qhpi_op_input(op, idx).op);
}

/** Get raw constant data pointer for a constant op. */
inline const void *constant_data(const QHPI_Op *op)
{
    return qhpi_op_constant_data(op);
}

/** True when the op's fully-qualified name equals @p name. */
inline bool op_is(const QHPI_Op *op, const char *name)
{
    return std::strcmp(qhpi_op_name(op), name) == 0;
}

/**
 * True when the op's name equals @p name AND it has exactly @p num_inputs
 * inputs. Fuses the common "is this the op I expect, and is its arity the
 * one my rewrite assumes" guard used before destructuring inputs.
 */
inline bool op_is(const QHPI_Op *op, const char *name, uint32_t num_inputs)
{
    return qhpi_op_num_inputs(op) == num_inputs && std::strcmp(qhpi_op_name(op), name) == 0;
}

/**
 * Read one integer element from a constant input tensor, regardless of its
 * declared element type (int32, or any narrower signed/unsigned/quantized
 * integer type — quantization parameters are not applied; the stored integer
 * is read as-is). Byte width is taken from qhpi_element_type_size(); signed
 * types are sign-extended, unsigned types are zero-extended.
 * @param op         op whose input is being read
 * @param input_idx  index of the input that holds the constant
 * @param elem_idx   linear element index into the constant data
 * @param out        element value (unchanged on failure)
 * @return false if the input is not a constant, its data pointer is null,
 *         or its element type is not an integer type of size 1/2/4 bytes.
 */
inline bool constval_int(const QHPI_Op *op, uint32_t input_idx, uint32_t elem_idx, int32_t &out)
{
    const QHPI_OpRef ref = qhpi_op_input(op, input_idx);
    if (!qhpi_op_is_constant(ref.op)) return false;
    const uint8_t *const data = static_cast<const uint8_t *>(qhpi_op_constant_data(ref.op));
    if (!data) return false;

    const QHPI_Element_Type type = output_type(ref.op);
    const bool is_supported = type == QHPI_QUINT8 || type == QHPI_QINT8 || type == QHPI_QUINT16 ||
                              type == QHPI_QINT16 || type == QHPI_INT32 || type == QHPI_QINT32;
    if (!is_supported) return false;
    const bool is_signed = type != QHPI_QUINT8 && type != QHPI_QUINT16;

    const uint32_t width = qhpi_element_type_size(type);
    const uint8_t *const elem = data + elem_idx * width;
    switch (width) {
    case 1:
        out = is_signed ? *reinterpret_cast<const int8_t *>(elem) : *elem;
        return true;
    case 2:
        out = is_signed ? *reinterpret_cast<const int16_t *>(elem) : *reinterpret_cast<const uint16_t *>(elem);
        return true;
    case 4:
        out = *reinterpret_cast<const int32_t *>(elem);
        return true;
    default:
        return false;
    }
}

/**
 * std::optional-returning sugar for constval_int(), for callers that prefer
 * to check-and-use in one expression rather than pass an out-param:
 *
 *   auto keep_dims = qhpi::constval_int_optional(op, KEEP_DIMS);
 *   if (!keep_dims || *keep_dims != 1) return op;
 *
 * @param elem_idx  linear element index into the constant data (default 0,
 *                  the common case of reading a scalar attribute constant).
 */
inline std::optional<int32_t> constval_int_optional(const QHPI_Op *op, uint32_t input_idx, uint32_t elem_idx = 0)
{
    int32_t out;
    if (!constval_int(op, input_idx, elem_idx, out)) return std::nullopt;
    return out;
}

} // namespace qhpi

#endif // DEFOPT2QHPI_H
