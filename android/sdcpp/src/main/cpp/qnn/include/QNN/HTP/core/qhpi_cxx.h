// ==============================================================================
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear
//
// ==============================================================================

#pragma once

#include "qhpi.h"

#include <array>
#include <cassert>
#include <deque>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace qhpi {

// Unscoped enum: kernel callbacks return uint32_t, so values must implicitly convert.
namespace Status {
enum Status {
    Success = QHPI_SUCCESS,
    Unsupported = QHPI_UNSUPPORTED,
    Error = QHPI_ERROR_FATAL,
    RegistrationWarning = QHPI_REGISTRATION_WARNING,
    RegistrationError = QHPI_REGISTRATION_ERROR
};
}

using Shape = QHPI_Shape;

// A single dimension of a Shape, as yielded by iterating over a Shape:
//
//   for (auto [d, size] : shape) { ... }
struct ShapeDim {
    uint32_t dim;
    uint32_t size;
};

struct ShapeIterator {
    const uint32_t *dims;
    uint32_t index;
    ShapeDim operator*() const { return {index, dims[index]}; }
    ShapeIterator &operator++()
    {
        ++index;
        return *this;
    }
    bool operator!=(const ShapeIterator &other) const { return index != other.index; }
};

} // namespace qhpi

// QHPI_Shape is a plain C type, so these must live in the global namespace
// for ADL to find them from range-for — including uses inside namespace qhpi
// itself, since these declarations precede them textually.
inline qhpi::ShapeIterator begin(const QHPI_Shape &shape)
{
    return {shape.dims, 0};
}
inline qhpi::ShapeIterator end(const QHPI_Shape &shape)
{
    return {shape.dims, shape.rank};
}

namespace qhpi {

enum class ElementType : uint32_t {
    QUINT8 = QHPI_QUINT8,
    QUINT16 = QHPI_QUINT16,
    QINT16 = QHPI_QINT16,
    FLOAT32 = QHPI_FLOAT32,
    INT32 = QHPI_INT32,
    QINT32 = QHPI_QINT32,
    QINT8 = QHPI_QINT8,
    FLOAT16 = QHPI_FLOAT16,
    INT64 = QHPI_INT64,
    ANY = QHPI_ELEMENT_TYPE_ANY
};

using QuantParameters = QHPI_Quant_Parameters;
using OutputDefinition = QHPI_OutputDef;
using Op = QHPI_Op;
using OperatorReference = QHPI_OpRef;

using TileShapeRequired = QHPI_TileShapeRequired;
using TileShapeLegalized = QHPI_TileShapeLegalized;
using BuildTileOfOperator = QHPI_BuildTileOfOp;

constexpr auto NO_TILING = QHPI_NO_TILING;

using RewriteOperatorFunction = QHPI_RewriteOpFunc;
using DimensionInfo = QHPI_DimInfo;
using ChunkedMemoryLayout = QHPI_ChunkedMemoryLayout;

enum class Layout : uint32_t {
    FLAT4 = QHPI_LAYOUT_FLAT_4,
    FLAT5 = QHPI_LAYOUT_FLAT_5,
    FLAT6 = QHPI_LAYOUT_FLAT_6,
    SINGULAR = QHPI_LAYOUT_SINGULAR,
    NCHW = QHPI_LAYOUT_NCHW,
    DEPTH32 = QHPI_LAYOUT_DEPTH_32,
    WEIGHTS_8X4 = QHPI_LAYOUT_WEIGHTS_8X4,
    CROUTON8 = QHPI_LAYOUT_CROUTON_8,
    CROUTON16 = QHPI_LAYOUT_CROUTON_16,
    CROUTON32 = QHPI_LAYOUT_CROUTON_32,
    CROUTON_4X1 = QHPI_LAYOUT_CROUTON_4X1,
    CROUTON_2X2 = QHPI_LAYOUT_CROUTON_2X2,
    WIDE_CROUTON8 = QHPI_LAYOUT_WIDE_CROUTON_8,
    WIDE_CROUTON32 = QHPI_LAYOUT_WIDE_CROUTON_32,
    WIDE_CROUTON_2X2 = QHPI_LAYOUT_WIDE_CROUTON_2X2,
    ANY = QHPI_LAYOUT_ANY
};

enum class MemoryLocation : uint8_t {
    DDR = QHPI_MEM_LOC_DDR_ONLY,
    TCM = QHPI_MEM_LOC_TCM_ONLY,
    ANY = QHPI_MEM_LOC_DDR_OR_TCM
};

enum class Storage : uint8_t {
    DIRECT = QHPI_STORAGE_DIRECT,
    INDIRECT = QHPI_STORAGE_INDIRECT,
    ANY = QHPI_STORAGE_DIRECT_OR_INDIRECT
};

class TensorSignature : public QHPI_Tensor_Signature_v1 {
  public:
    constexpr TensorSignature(ElementType element_type, Layout layout, MemoryLocation memory_location,
                              Storage storage = Storage::DIRECT)
        : QHPI_Tensor_Signature_v1{.element_type = static_cast<uint32_t>(element_type),
                                   .layout = static_cast<uint32_t>(layout),
                                   .storage = static_cast<uint8_t>(storage),
                                   .mem_placement = static_cast<uint8_t>(memory_location)}
    {
    }
};

namespace tensor_signature {
constexpr TensorSignature ANY = {ElementType::ANY, Layout::ANY, MemoryLocation::ANY, Storage::ANY};
constexpr TensorSignature FLOAT16_FLAT4_DDR = {ElementType::FLOAT16, Layout::FLAT4, MemoryLocation::DDR};
constexpr TensorSignature FLOAT32_FLAT4_DDR = {ElementType::FLOAT32, Layout::FLAT4, MemoryLocation::DDR};
constexpr TensorSignature INT32_FLAT4_DDR = {ElementType::INT32, Layout::FLAT4, MemoryLocation::DDR};
constexpr TensorSignature INT32_FLAT4_TCM = {ElementType::INT32, Layout::FLAT4, MemoryLocation::TCM};
constexpr TensorSignature INT32_CROUTON32_DDR = {ElementType::INT32, Layout::CROUTON32, MemoryLocation::DDR};
constexpr TensorSignature INT32_CROUTON32_INDIRECT_DDR = {ElementType::INT32, Layout::CROUTON32, MemoryLocation::DDR,
                                                          Storage::INDIRECT};
constexpr TensorSignature QUINT8_FLAT4_DDR = {ElementType::QUINT8, Layout::FLAT4, MemoryLocation::DDR};
constexpr TensorSignature QUINT16_FLAT4_DDR = {ElementType::QUINT16, Layout::FLAT4, MemoryLocation::DDR};
} // namespace tensor_signature

using Tensor = QHPI_Tensor;

inline Shape tensor_shape(const Tensor *tensor)
{
    return qhpi_tensor_shape(tensor);
}
inline Shape tensor_padded_shape(const Tensor *tensor)
{
    return qhpi_tensor_padded_shape(tensor);
}
template <typename T> inline T *tensor_data(Tensor *tensor)
{
    return static_cast<T *>(qhpi_tensor_raw_data(tensor));
}

// Unpacks a fixed-arity array of tensor pointers into a std::array, for use
// with structured bindings:
//
//   auto [input, axis_tensor] = unpack<2>(inputs);
//   auto [output] = unpack<1>(outputs);
//
// N is the kernel's declared input/output count (its TensorSignature arity),
// not derivable from the raw pointer and so given explicitly.
template <unsigned N, typename T> inline std::array<T, N> unpack(const T *ptrs)
{
    std::array<T, N> out;
    for (unsigned i = 0; i < N; ++i)
        out[i] = ptrs[i];
    return out;
}

// Reads the first element of a tensor's raw data as T. For the common case
// of a rank-4 (1,1,1,1) scalar input (e.g. an "axis" or "keep_dims" operand).
template <typename T> inline T tensor_scalar(const Tensor *tensor)
{
    return *static_cast<const T *>(qhpi_tensor_raw_data(tensor));
}

// Builds an OutputDefinition for a non-quantized intermediate tensor -- e.g.
// a scratch output added by an early_rewrite whose backing memory is owned by
// the graph's tensor allocator rather than allocated at execute time.
inline OutputDefinition scratch_output(ElementType type, const Shape &shape)
{
    return {.type = static_cast<uint32_t>(type), .quant_parameters = QHPI_Quant_Parameters_Invalid, .shape = shape};
}

// Resolves a possibly-negative axis (QNN convention: negative counts back
// from the end) against a tensor rank. Returns std::nullopt if out of range
// after normalization.
inline std::optional<uint8_t> normalize_axis(int32_t axis, uint32_t rank)
{
    if (axis < 0) axis += static_cast<int32_t>(rank);
    if (axis < 0 || static_cast<uint32_t>(axis) >= rank) return std::nullopt;
    return static_cast<uint8_t>(axis);
}

// Fixed-capacity storage for a ChunkedMemoryLayout, sized to hold the
// largest descriptor qhpi_chunked_layout() can produce. Inherits rank
// and dim_info[0] from the base struct; extra_dim_info supplies the
// remaining backing storage for dim_info[1..]. Sized to the full
// QHPI_MAX_MEMORY_DIM_INFO_SIZE, not one less, because tail padding after
// the base struct's inline dim_info[1] (12 bytes, not 10) already eats one
// slot's worth of space.
struct ChunkedLayoutStorage : ChunkedMemoryLayout {
    QHPI_DimInfo extra_dim_info[QHPI_MAX_MEMORY_DIM_INFO_SIZE];
};
static_assert(sizeof(ChunkedLayoutStorage) >= QHPI_MAX_MEMORY_LAYOUT_SIZE);

// Fetches the chunked-memory-layout descriptor for a tensor. Returns
// std::nullopt if the tensor's layout has no chunked descriptor
// (Layout::ANY, SHAPE_ONLY, CUSTOM).
inline std::optional<ChunkedLayoutStorage> chunked_layout(const Tensor *tensor)
{
    ChunkedLayoutStorage layout{};
    if (qhpi_chunked_layout(qhpi_tensor_layout(tensor), &layout, sizeof(layout)) != QHPI_SUCCESS) return std::nullopt;
    return layout;
}

using DimInfoBounds = std::array<uint32_t, QHPI_MAX_MEMORY_DIM_INFO_SIZE>;
using Coordinates = std::array<uint32_t, QHPI_MAX_RANK>;

// A single dim_info[] entry of a ChunkedMemoryLayout, as yielded by
// iterating over the layout:
//
//   for (auto [index, info] : layout) { ... }
struct ChunkedLayoutIterator {
    const QHPI_DimInfo *dim_info;
    uint32_t index;
    std::pair<uint32_t, QHPI_DimInfo> operator*() const { return {index, dim_info[index]}; }
    ChunkedLayoutIterator &operator++()
    {
        ++index;
        return *this;
    }
    bool operator!=(const ChunkedLayoutIterator &other) const { return index != other.index; }
};

} // namespace qhpi

// QHPI_ChunkedMemoryLayout is a plain C type, so these must live in the
// global namespace for ADL to find them from range-for — including uses
// inside namespace qhpi itself, since these declarations precede them
// textually.
inline qhpi::ChunkedLayoutIterator begin(const QHPI_ChunkedMemoryLayout &layout)
{
    return {layout.dim_info, 0};
}
inline qhpi::ChunkedLayoutIterator end(const QHPI_ChunkedMemoryLayout &layout)
{
    return {layout.dim_info, layout.dim_info_size};
}

namespace qhpi {

// Computes the iteration bound for each dim_info level of a chunked memory
// layout: chunk_size if nonzero, else padded_dim / (product of nonzero
// chunk_sizes for that dim). Only the first layout.dim_info_size entries
// of the returned array are valid.
inline DimInfoBounds dim_info_bounds(const ChunkedMemoryLayout &layout, const Shape &padded_shape)
{
    Coordinates inner_product{};
    for (auto [d, _] : padded_shape)
        inner_product[d] = 1;
    for (auto [i, info] : layout)
        if (info.chunk_size > 0) inner_product[info.dim] *= info.chunk_size;

    DimInfoBounds bound{};
    for (auto [i, info] : layout)
        bound[i] = info.chunk_size > 0 ? info.chunk_size : padded_shape.dims[info.dim] / inner_product[info.dim];
    return bound;
}

// Iterates over every element of a tensor in physical memory order, given its
// ChunkedMemoryLayout descriptor and padded shape. Call advance() between
// elements; it returns false after the last element. Use logical_coordinates()
// at each step to recover logical dimension indices.
class LayoutAgnosticOdometer {
  public:
    LayoutAgnosticOdometer(const ChunkedLayoutStorage &layout, const Shape &padded_shape)
        : m_layout{layout}, m_bounds{dim_info_bounds(layout, padded_shape)},
          m_elements_per_block{compute_elements_per_block(m_bounds, layout)}
    {
    }

    // Advance to the next element. Returns false after the last element.
    bool advance()
    {
        ++m_physical_index;
        for (uint32_t i = m_layout.dim_info_size; i-- > 0;) {
            if (++m_counter[i] < m_bounds[i]) return true;
            m_counter[i] = 0;
        }
        return false;
    }

    // Logical coordinates of all dimensions at the current position.
    // Entries at index >= padded rank are unspecified.
    Coordinates logical_coordinates() const
    {
        Coordinates coordinates{};
        for (auto [i, info] : m_layout)
            coordinates[info.dim] = coordinates[info.dim] * m_bounds[i] + m_counter[i];
        return coordinates;
    }

    // Physical (memory walk order) index of the current element. Pass to
    // StorageAgnosticAccessor::read()/ref(). Starts at 0 and advances by one
    // per advance() call, in lockstep with logical_coordinates().
    uint32_t physical_index() const { return m_physical_index; }

    // Number of elements per block. Pass to StorageAgnosticAccessor's constructor.
    uint32_t elements_per_block() const { return m_elements_per_block; }

  private:
    static uint32_t compute_elements_per_block(const DimInfoBounds &bounds, const ChunkedMemoryLayout &layout)
    {
        uint32_t product = 1;
        for (auto [i, info] : layout)
            if (info.chunk_size > 0) product *= bounds[i];
        return product;
    }

    ChunkedLayoutStorage m_layout;
    DimInfoBounds m_bounds;
    uint32_t m_elements_per_block;
    uint32_t m_counter[QHPI_MAX_MEMORY_DIM_INFO_SIZE] = {};
    uint32_t m_physical_index = 0;
};

// True when every coordinate lies within its dimension's logical extent.
// Used by layout-agnostic kernels to skip padding slots surfaced by an
// odometer walking a padded shape.
inline bool in_bounds(const Coordinates &coordinates, const Shape &shape)
{
    for (auto [dim, size] : shape)
        if (coordinates[dim] >= size) return false;
    return true;
}

// Product of every dimension of shape except reduce_dim. For a reduction
// kernel, this is the number of independent output slots (one per unique
// combination of non-reduction coordinates); size the scratch that tracks
// per-slot state (running max, running sum, ...) to this value.
inline uint32_t outer_size(const Shape &shape, uint32_t reduce_dim)
{
    uint32_t product = 1;
    for (auto [dim, size] : shape)
        if (dim != reduce_dim) product *= size;
    return product;
}

// Row-major linearization of coordinates across every dimension of shape
// except reduce_dim. Companion to outer_size(): a scratch buffer sized by
// outer_size() is indexed by outer_index() at each element visited.
inline uint32_t outer_index(const Coordinates &coordinates, const Shape &shape, uint32_t reduce_dim)
{
    uint32_t index = 0;
    for (auto [dim, size] : shape)
        if (dim != reduce_dim) index = index * size + coordinates[dim];
    return index;
}

// Per-element read/write over a tensor, abstracting direct and indirect
// storage. sizeof(T) must match the tensor's element size; on mismatch the
// accessor is left invalid rather than aborting, so callers must check
// valid() before calling read() or ref(). Quantized types are accessed via
// their plain C counterparts (QUINT8→uint8_t, QINT8→int8_t, QUINT16→uint16_t,
// QINT16→int16_t, QINT32→int32_t); the Q prefix indicates associated quant
// parameters, not a different storage format.
template <typename T> class StorageAgnosticAccessor {
  public:
    StorageAgnosticAccessor(const Tensor *tensor, uint32_t elements_per_block)
        : m_elements_per_block{elements_per_block}
    {
        if (sizeof(std::remove_cv_t<T>) != qhpi_element_type_size(qhpi_tensor_type(tensor))) return;
        if (qhpi_tensor_is_indirect(tensor))
            m_table = qhpi_tensor_block_table(tensor);
        else
            m_raw = static_cast<T *>(qhpi_tensor_raw_data(tensor));
    }

    bool valid() const { return m_raw != nullptr || m_table != nullptr; }

    T read(uint32_t element) const { return element_ref(element); }
    T &ref(uint32_t element) const { return element_ref(element); }

  private:
    T &element_ref(uint32_t element) const
    {
        if (m_raw) return m_raw[element];
        return static_cast<T *>(m_table[element / m_elements_per_block])[element % m_elements_per_block];
    }

    T *m_raw = nullptr;
    void **m_table = nullptr;
    uint32_t m_elements_per_block;
};

using KernelStatus = QHPI_StatusCode;
using RuntimeHandle = QHPI_RuntimeHandle;
using PluginFunction = QHPI_Plugin_Function;
using PrecomputeFunction = QHPI_Precompute_Function;
using PluginPrecomputedFunction = QHPI_Plugin_Precomputed_Function;
using KernelPredicate = QHPI_Kernel_Predicate;
using CostFunction = QHPI_Cost_Function;

// Bitmask, not a single-value enum: callers OR values together (e.g. HVX | Main).
using Resource = QHPI_Resource;
namespace resource {
constexpr auto Main = QHPI_RESOURCE_MAIN;
constexpr auto HVX = QHPI_RESOURCE_HVX;
constexpr auto HMX = QHPI_RESOURCE_HMX;
constexpr auto Exclusive = QHPI_RESOURCE_EXCLUSIVE;
} // namespace resource

enum class SourceDestructive : uint8_t { No = 0, Yes = 1 };
enum class Multithreaded : uint8_t { No = 0, Yes = 1 };
enum class VariableInputs : uint8_t { No = 0, Yes = 1 };
enum class VariableOutputs : uint8_t { No = 0, Yes = 1 };

struct KernelOptions {
    // Dispatch.
    Resource resources = resource::HVX;
    Multithreaded multithreaded = Multithreaded::No;
    uint32_t sync_block_size = 0;
    // Behavior.
    SourceDestructive source_destructive = SourceDestructive::No;
    VariableInputs variable_inputs = VariableInputs::No;
    VariableOutputs variable_outputs = VariableOutputs::No;
    // Hooks.
    CostFunction cost_function = nullptr;
    KernelPredicate predicate = nullptr;
};

class Kernel {
  public:
    // Plain kernel.
    Kernel(std::string name, PluginFunction function, std::vector<TensorSignature> inputs,
           std::vector<TensorSignature> outputs, KernelOptions options = {})
        : Kernel(std::move(name), function, 0, nullptr, nullptr, std::move(inputs), std::move(outputs), options)
    {
    }

    // Plain kernel, single output.
    Kernel(std::string name, PluginFunction function, std::vector<TensorSignature> inputs, TensorSignature output,
           KernelOptions options = {})
        : Kernel(std::move(name), function, std::move(inputs), std::vector<TensorSignature>{output}, options)
    {
    }

    // Kernel with precomputed data: precompute_function fills a buffer of precomputed_data_size,
    // which plugin_precomputed_function then receives at execution time.
    Kernel(std::string name, uint32_t precomputed_data_size, PrecomputeFunction precompute_function,
           PluginPrecomputedFunction plugin_precomputed_function, std::vector<TensorSignature> inputs,
           std::vector<TensorSignature> outputs, KernelOptions options = {})
        : Kernel(std::move(name), nullptr, precomputed_data_size, precompute_function, plugin_precomputed_function,
                 std::move(inputs), std::move(outputs), options)
    {
    }

    // Precomputed-data kernel, single output.
    Kernel(std::string name, uint32_t precomputed_data_size, PrecomputeFunction precompute_function,
           PluginPrecomputedFunction plugin_precomputed_function, std::vector<TensorSignature> inputs,
           TensorSignature output, KernelOptions options = {})
        : Kernel(std::move(name), precomputed_data_size, precompute_function, plugin_precomputed_function,
                 std::move(inputs), std::vector<TensorSignature>{output}, options)
    {
    }

    // The C API takes non-const pointers to the signature arrays and kernel array but only reads
    // through them, so we cast away const on our own storage to satisfy the pointer types.
    QHPI_Kernel_v1 c_struct() const
    {
        auto &self = *const_cast<Kernel *>(this);
        auto out = c_;
        out.function_name = self.m_name.c_str();
        out.input_signature = self.m_inputs.data();
        out.output_signature = self.m_outputs.data();
        return out;
    }

  private:
    Kernel(std::string name, PluginFunction function, uint32_t precomputed_data_size,
           PrecomputeFunction precompute_function, PluginPrecomputedFunction plugin_precomputed_function,
           std::vector<TensorSignature> inputs, std::vector<TensorSignature> outputs, KernelOptions options = {})
        : m_name(std::move(name)), m_inputs(std::move(inputs)), m_outputs(std::move(outputs)),
          c_{.function_name = nullptr,
             .function = function,
             .resources = static_cast<uint8_t>(options.resources),
             .source_destructive = static_cast<uint8_t>(options.source_destructive),
             .multithreaded = static_cast<uint8_t>(options.multithreaded),
             .variable_inputs = static_cast<uint8_t>(options.variable_inputs),
             .variable_outputs = static_cast<uint8_t>(options.variable_outputs),
             .min_inputs = static_cast<uint32_t>(m_inputs.size()),
             .input_signature = nullptr,
             .min_outputs = static_cast<uint32_t>(m_outputs.size()),
             .output_signature = nullptr,
             .cost_function = options.cost_function,
             .sync_block_size = options.sync_block_size,
             .precomputed_data_size = precomputed_data_size,
             .do_precomputation_function = precompute_function,
             .function_with_precomputed_data = plugin_precomputed_function,
             .predicate = options.predicate,
             .Reserved_1 = nullptr,
             .Reserved_2 = nullptr,
             .Reserved_3 = nullptr,
             .Reserved_4 = nullptr}
    {
    }

    std::string m_name;
    std::vector<TensorSignature> m_inputs;
    std::vector<TensorSignature> m_outputs;
    // Self-referential pointer fields (function_name, input_signature, output_signature) are filled
    // in by c_struct() on read, not stored here, so default copy/move stay correct after resize.
    QHPI_Kernel_v1 c_;
};

namespace internal {
inline std::vector<QHPI_Kernel_v1> kernels_to_c(const std::vector<Kernel> &kernels)
{
    std::vector<QHPI_Kernel_v1> out;
    out.reserve(kernels.size());
    for (const auto &k : kernels)
        out.push_back(k.c_struct());
    return out;
}
} // namespace internal

struct OperatorInfoOptions {
    RewriteOperatorFunction early_rewrite = nullptr;
    RewriteOperatorFunction late_rewrite = nullptr;
    TileShapeRequired tile_shape_required = nullptr;
    TileShapeLegalized tile_shape_legalized = nullptr;
    uint32_t tile_output = 0;
    BuildTileOfOperator build_tile = nullptr;
};

class OperatorInfo {
  public:
    OperatorInfo(const OperatorInfo &other)
        : m_name(other.m_name), m_kernels(other.m_kernels), m_c_kernels(internal::kernels_to_c(m_kernels)), c_(other.c_)
    {
    }
    OperatorInfo(OperatorInfo &&) noexcept = default;
    OperatorInfo &operator=(const OperatorInfo &other) { return *this = OperatorInfo(other); }
    OperatorInfo &operator=(OperatorInfo &&) noexcept = default;

    OperatorInfo(std::string name, std::vector<Kernel> kernels, OperatorInfoOptions options = {})
        : m_name(std::move(name)), m_kernels(std::move(kernels)), m_c_kernels(internal::kernels_to_c(m_kernels)),
          c_{.name = nullptr,
             .num_kernels = 0,
             .kernels = nullptr,
             .early_rewrite = options.early_rewrite,
             .shape_required = options.tile_shape_required,
             .shape_legalized = options.tile_shape_legalized,
             .tile_output = options.tile_output,
             .build_tile = options.build_tile,
             .late_rewrite = options.late_rewrite,
             .Reserved_1 = nullptr,
             .Reserved_2 = nullptr,
             .Reserved_3 = nullptr,
             .Reserved_4 = nullptr}
    {
    }

    // Rewrite-only operator: no kernel implementations. Pass options as second arg with designated
    // init (e.g. `{.early_rewrite = rename_x}`); a bare `{}` is ambiguous with the empty-kernels ctor.
    OperatorInfo(std::string name, OperatorInfoOptions options) : OperatorInfo(std::move(name), {}, options) {}

    // The C API takes a non-const pointer to the kernels array but only reads through it, so we
    // cast away const on our own storage to satisfy the pointer type.
    QHPI_OpInfo_v1 c_struct() const
    {
        auto &self = *const_cast<OperatorInfo *>(this);
        auto out = c_;
        out.name = self.m_name.c_str();
        out.num_kernels = static_cast<uint32_t>(self.m_c_kernels.size());
        out.kernels = self.m_c_kernels.empty() ? nullptr : self.m_c_kernels.data();
        return out;
    }

  private:
    std::string m_name;
    std::vector<Kernel> m_kernels;
    std::vector<QHPI_Kernel_v1> m_c_kernels;
    // Self-referential pointer fields (name, kernels) and size are filled in by c_struct() on read,
    // not stored here, so default copy/move stay correct after resize.
    QHPI_OpInfo_v1 c_;
};

namespace internal {
inline std::vector<OperatorInfo> &operator_registry()
{
    static auto r = std::vector<OperatorInfo>{};
    return r;
}

struct Package {
    std::string name;
    std::vector<OperatorInfo> operator_infos;
    std::vector<QHPI_OpInfo_v1> c_operator_infos;
};
// Container must be pointer-stable across emplace_back: the runtime holds pointers into
// Package::name and Package::c_operator_infos for the program's lifetime via qhpi_register_ops_v1.
inline auto all_packages = std::deque<Package>{};

inline const char *init(std::string_view package)
{
    auto &p = all_packages.emplace_back();
    p.name = package;
    p.operator_infos = std::move(operator_registry());
    p.c_operator_infos.reserve(p.operator_infos.size());
    for (const auto &op : p.operator_infos)
        p.c_operator_infos.push_back(op.c_struct());
    qhpi_register_ops_v1(static_cast<uint32_t>(p.c_operator_infos.size()), p.c_operator_infos.data(), p.name.c_str());
    return p.name.c_str();
}
} // namespace internal

// Anchor the return value at file scope to register operators with the package.
inline bool register_operators(OperatorInfo operator_info)
{
    internal::operator_registry().push_back(std::move(operator_info));
    return true;
}
inline bool register_operators(std::vector<OperatorInfo> operator_infos)
{
    for (auto &op : operator_infos)
        register_operators(std::move(op));
    return true;
}

} // namespace qhpi

// Defines the extern "C" qhpi_init entry point. Invoke once at file scope (not inside a namespace,
// where extern "C" linkage would be rejected). All operators must be registered via
// register_operators.
#define QHPI_REGISTER_PACKAGE(package_name)                                                                            \
    extern "C" const char *qhpi_init()                                                                                 \
    {                                                                                                                  \
        return qhpi::internal::init((package_name));                                                                   \
    }
