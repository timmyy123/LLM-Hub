// ==============================================================================
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear
//
// ==============================================================================

/**
 * @file qhpi.h
 * @brief Qualcomm HTP Plugin Interface (QHPI).
 *
 * QHPI is the C API through which plugin operators integrate with
 * the graph compiler. It provides facilities for:
 *   - Registering custom operator packages with kernel implementations.
 *   - Inspecting and transforming the operator graph during preparation
 *     (rewriting, tiling).
 *   - Accessing tensor data and metadata at execution time.
 *
 * All graph-manipulation functions (qhpi_op_create, qhpi_op_slice, etc.)
 * are available only during the prepare phase. Tensor accessor functions
 * (qhpi_tensor_shape, qhpi_tensor_raw_data, etc.) are available at
 * execution time inside plugin kernel callbacks.
 */

#ifndef QHPI_QHPI
#define QHPI_QHPI 1

#include <float.h>
#include <limits.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
#define HEXNN_QHPI_NULL nullptr
#else
#define HEXNN_QHPI_NULL NULL
#endif

/** @brief Maximum tensor rank supported by QHPI. */
#define QHPI_MAX_RANK 8

/** @brief Default package name used when none is specified. */
#define QHPI_DEFAULT_PACKAGE "q"

/** @brief Maximum number of QHPI_DimInfo entries in a chunked memory layout. */
#define QHPI_MAX_MEMORY_DIM_INFO_SIZE 9

/** @brief Maximum byte size needed to store a QHPI_ChunkedMemoryLayout with all dim_info entries. */
#define QHPI_MAX_MEMORY_LAYOUT_SIZE                                                                                    \
    (sizeof(QHPI_ChunkedMemoryLayout) + (sizeof(QHPI_DimInfo) * QHPI_MAX_MEMORY_DIM_INFO_SIZE))

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Status codes returned by QHPI functions and plugin callbacks.
 */
enum {
    QHPI_SUCCESS = 0, /**< Operation completed successfully. */
    QHPI_UNSUPPORTED = 1, /**< Requested feature or option is not supported. */
    QHPI_ERROR_FATAL = 2, /**< A fatal error occurred. */
    QHPI_REGISTRATION_WARNING = 3, /**< Registration succeeded with warnings. */
    QHPI_REGISTRATION_ERROR = 4, /**< Registration failed. */
};
typedef uint32_t QHPI_StatusCode;

/**
 * @brief A small fixed-capacity vector of dimension sizes.
 *
 * Represents the shape of a tensor. The effective length is given by @c rank;
 * only the first @c rank entries of @c dims are meaningful. Each meaningful
 * entry is a positive dimension size.
 */
typedef struct {
    uint32_t rank; /**< Number of valid entries in @c dims. */
    uint32_t dims[QHPI_MAX_RANK]; /**< Dimension sizes (first @c rank entries are valid). */
} QHPI_Shape;

/** @brief Sentinel value indicating an invalid rank. */
static const uint32_t QHPI_Rank_Invalid = UINT32_MAX;

/** @brief Sentinel value indicating an invalid shape. */
static const QHPI_Shape QHPI_Shape_Invalid = {QHPI_Rank_Invalid, {0}};

/**
 * @brief Element data types for tensor values.
 *
 * Types prefixed with @c Q are quantized types that use QHPI_Quant_Parameters
 * to define the mapping between stored integer values and real numbers.
 */
enum {
    QHPI_UNKNOWN = 0, /**< Unknown or unspecified type. */
    QHPI_QUINT8 = 1, /**< Quantized unsigned 8-bit integer. */
    QHPI_QUINT16 = 2, /**< Quantized unsigned 16-bit integer. */
    QHPI_QINT16 = 3, /**< Quantized signed 16-bit integer. */
    QHPI_FLOAT32 = 4, /**< 32-bit float. */
    QHPI_INT32 = 5, /**< Signed 32-bit integer. */
    QHPI_QINT32 = 6, /**< Quantized signed 32-bit integer. */
    QHPI_QINT8 = 7, /**< Quantized signed 8-bit integer. */
    QHPI_FLOAT16 = 8, /**< 16-bit float. */
    QHPI_INT64 = 9, /**< Signed 64-bit integer. */
    QHPI_ELEMENT_TYPE_ANY = 240, /**< Wildcard matching any element type (for signatures). */
    QHPI_ELEMENT_TYPE_INVALID = 241, /**< Pseudo-type indicating a preparation error. */
};
typedef uint32_t QHPI_Element_Type;

/**
 * @brief Return the size in bytes of a single element of the given type.
 *
 * @param type The element type to query.
 * @return Size in bytes, or 0 for unsupported types (e.g. QHPI_UNKNOWN,
 *         QHPI_ELEMENT_TYPE_INVALID).
 */
uint32_t qhpi_element_type_size(QHPI_Element_Type type);

/**
 * @brief Quantization parameters mapping stored integers to real values.
 *
 * The real value corresponding to a stored integer @c x is:
 *   @code real_value = (x - zero_offset) * stepsize @endcode
 *
 * Only meaningful for quantized element types (QHPI_QUINT8, QHPI_QINT8, etc.).
 */
typedef struct {
    int32_t zero_offset; /**< Integer value that maps to real zero. */
    float stepsize; /**< Scale factor (real units per integer step). */
} QHPI_Quant_Parameters;

/** @brief Sentinel value indicating invalid quantization parameters. */
static const QHPI_Quant_Parameters QHPI_Quant_Parameters_Invalid = {INT32_MAX, FLT_MAX};

/**
 * @brief Properties of an output tensor for an operator.
 *
 * Describes the element type, quantization parameters, and shape of one
 * output produced by an operator. Used when creating new operators and
 * when querying existing operator outputs.
 */
typedef struct {
    QHPI_Element_Type type; /**< Element data type. */
    QHPI_Quant_Parameters quant_parameters; /**< Quantization parameters (meaningful for quantized types only). */
    QHPI_Shape shape; /**< Tensor shape (rank 0 is valid for scalars). */
} QHPI_OutputDef;

/** @brief Sentinel value indicating an invalid output tensor. */
static const QHPI_OutputDef QHPI_OutputDef_Invalid = {QHPI_ELEMENT_TYPE_INVALID, QHPI_Quant_Parameters_Invalid,
                                                      QHPI_Shape_Invalid};

/**
 * @brief Opaque handle to an operator node in the graph.
 *
 * QHPI_Op pointers are used during the prepare phase to inspect and
 * transform the operator graph. They are not valid across different
 * graph instances or after preparation completes.
 */
typedef struct QHPI_Op QHPI_Op;

/**
 * @brief Reference to a specific output of an operator.
 *
 * Since operators may produce multiple outputs, an OpRef pairs an operator
 * pointer with an output index. This is the primary way tensors are
 * represented during graph optimization.
 */
typedef struct {
    const QHPI_Op *op; /**< The operator that produces the tensor. */
    uint32_t output_number; /**< Zero-based index of the output on that operator. */
} QHPI_OpRef;

/** @brief Sentinel value indicating an invalid output. */
static const QHPI_OpRef QHPI_OpRef_Invalid = {HEXNN_QHPI_NULL, UINT32_MAX};

/**
 * @brief Construct a QHPI_OpRef from an operator pointer and output index.
 *
 * @param op         Pointer to the source operator.
 * @param out_number Zero-based output index on @p op.
 * @return A QHPI_OpRef referencing the specified output.
 */
inline QHPI_OpRef qhpi_op_reference(const QHPI_Op *op, uint32_t out_number)
{
    QHPI_OpRef result;
    result.op = op;
    result.output_number = out_number;
    return result;
}

/**
 * @brief Return the name of an operator.
 *
 * The returned string includes the package prefix separated by "::"
 * (e.g. "my_package::MyOp"). Internal operators (those in the default
 * "q" package that are not QNN ops) return "<internal>".
 *
 * @param op Pointer to the operator to query.
 * @return Null-terminated operator name. The pointer remains valid for
 *         the lifetime of the graph.
 */
const char *qhpi_op_name(const QHPI_Op *op);

/**
 * @brief Return the number of inputs to an operator.
 *
 * @param op Pointer to the operator to query.
 * @return Number of input tensors.
 */
uint32_t qhpi_op_num_inputs(const QHPI_Op *op);

/**
 * @brief Return a reference to a specific input of an operator.
 *
 * The returned QHPI_OpRef identifies the upstream operator and output
 * index that feeds this input.
 *
 * @param op    Pointer to the operator to query.
 * @param index Zero-based input index. Must be less than qhpi_op_num_inputs().
 * @return QHPI_OpRef for the specified input, or an error reference if
 *         @p index is out of range.
 */
QHPI_OpRef qhpi_op_input(const QHPI_Op *op, uint32_t index);

/**
 * @brief Return the number of outputs from an operator.
 *
 * @param op Pointer to the operator to query.
 * @return Number of output tensors.
 */
uint32_t qhpi_op_num_outputs(const QHPI_Op *op);

/**
 * @brief Return the output definition for a specific output of an operator.
 *
 * @param op       Pointer to the operator to query.
 * @param position Zero-based output index. Must be less than qhpi_op_num_outputs().
 * @return QHPI_OutputDef describing the output's type, quantization, and shape.
 *         If @p position is invalid, the returned OutputDef has type QHPI_ELEMENT_TYPE_INVALID.
 */
QHPI_OutputDef qhpi_op_output(const QHPI_Op *op, uint32_t position);

/**
 * @brief Test whether an operator is a constant.
 *
 * Constant operators hold fixed data (weights, biases, etc.) that does not
 * change between inferences.
 *
 * @param op Pointer to the operator to query.
 * @return Non-zero if @p op is a constant, 0 otherwise.
 */
uint32_t qhpi_op_is_constant(const QHPI_Op *op);

/**
 * @brief Return a pointer to the raw data of a constant operator.
 *
 * Only valid when qhpi_op_is_constant() returns non-zero. The data is
 * laid out in the element type indicated by the operator's output definition.
 *
 * @param op Pointer to a constant operator.
 * @return Pointer to the constant data, or NULL if @p op is not a constant.
 */
const void *qhpi_op_constant_data(const QHPI_Op *op);

/**
 * @brief Create a new constant operator in the graph.
 *
 * The new constant holds a copy of the supplied data. The byte size
 * @p size must equal the product of the output shape dimensions multiplied
 * by the element size of the output type.
 *
 * @param root   An existing operator used to identify the graph context.
 * @param output Output definition (type, quantization, shape) for the constant.
 * @param size   Size of @p data in bytes.
 * @param data   Pointer to the constant data to copy.
 * @return Pointer to the new constant operator, or an error node if
 *         validation fails (e.g. size mismatch).
 */
const QHPI_Op *qhpi_op_create_constant(const QHPI_Op *root, const QHPI_OutputDef *output, uint32_t size,
                                       const void *data);

/**
 * @brief Create a shape-only operator encoding dimension information.
 *
 * The resulting operator carries only shape metadata with no tensor data.
 * This is useful for operations that need explicit shape parameters
 * (e.g. reshape targets, tiling specifications).
 *
 * @param root  An existing operator used to identify the graph context.
 * @param shape The shape to encode.
 * @return QHPI_OpRef referencing the new shape operator.
 */
QHPI_OpRef qhpi_op_create_shape(const QHPI_Op *root, const QHPI_Shape *shape);

/**
 * @brief Create a new operator in the graph.
 *
 * Inserts a new operator node with the given name, inputs, and output
 * definitions. The @p root parameter associates the new operator with
 * the correct graph context.
 *
 * @param root        An existing operator used to identify the graph context.
 * @param name        Null-terminated operator name, including the "package::"
 *                    prefix (e.g. "my_package::MyOp").
 * @param num_inputs  Number of input tensors.
 * @param inputs      Array of @p num_inputs QHPI_OpRef structures, each
 *                    referencing an output of a preceding operator.
 * @param num_outputs Number of output tensors.
 * @param outputs     Array of @p num_outputs QHPI_OutputDef structures
 *                    describing each output's type, quantization, and shape.
 * @return Pointer to the newly created operator, or an error node if
 *         a problem is detected.
 */
const QHPI_Op *qhpi_op_create(const QHPI_Op *root, const char *name, uint32_t num_inputs, const QHPI_OpRef *inputs,
                              uint32_t num_outputs, const QHPI_OutputDef *outputs);

/**
 * @brief Retrieve the value of an integer runtime option.
 *
 * Allows plugin operators to query integer-valued configuration options.
 * The special option names "tcm_size" and "tcm_size_for_tiling" return
 * the hardware VTCM size and the VTCM size available for tiling,
 * respectively.
 *
 * @param op    An operator providing the graph context.
 * @param name  Null-terminated name of the option to query.
 * @param[out] value  Receives the integer value if the option is found.
 * @return @c QHPI_SUCCESS if the option was found, or a non-zero status code
 *         (e.g. @c QHPI_UNSUPPORTED) if the option does not exist.
 */
QHPI_StatusCode qhpi_option_int(const QHPI_Op *op, const char *name, int32_t *value);

/**
 * @brief Return the number of consumers of an operator's output.
 *
 * Counts how many downstream operators use the output(s) of @p op as
 * an input.
 *
 * @param op Pointer to the operator to query.
 * @return Number of consumer operators.
 */
uint32_t qhpi_op_num_uses(const QHPI_Op *op);

/**
 * @brief Test whether an operator produces input for a named consumer.
 *
 * Checks if any immediate consumer of @p op has the operator name @p name.
 *
 * @param op   Pointer to the operator to query.
 * @param name Null-terminated operator name to search for among consumers.
 * @return Non-zero if @p op feeds an operator named @p name, 0 otherwise.
 */
uint32_t qhpi_op_producer_for(const QHPI_Op *op, const char *name);

/**
 * @brief Create a multi-output operator that copies each input as a
 *        separate output.
 *
 * The resulting operator has @p num_inputs outputs, where output @e i is
 * a copy of input @e i. This is useful for broadcasting a set of tensors
 * to multiple consumers.
 *
 * @param num_inputs Number of input (and output) tensors. Must be > 0.
 * @param inputs     Array of @p num_inputs QHPI_OpRef structures.
 * @return Pointer to the new multi-output operator, or NULL if
 *         @p num_inputs is 0.
 */
const QHPI_Op *qhpi_op_create_multi_output(uint32_t num_inputs, const QHPI_OpRef *inputs);

/**
 * @brief Create a slice of a tensor.
 *
 * Produces a sub-tensor starting at @p start with size @p extent along
 * each dimension. Both shapes must have the same rank as the input tensor.
 * For each dimension @e i, @c start->dims[i] + extent->dims[i] must not
 * exceed the input dimension size.
 *
 * @param ref    Reference to the tensor to slice.
 * @param start  Per-dimension start offsets.
 * @param extent Per-dimension sizes of the slice.
 * @return QHPI_OpRef referencing the slice result, or an error reference
 *         if validation fails.
 */
QHPI_OpRef qhpi_op_slice(QHPI_OpRef ref, const QHPI_Shape *start, const QHPI_Shape *extent);

/**
 * @brief Concatenate multiple tensors along a given axis.
 *
 * All inputs must have the same rank, element type, and quantization
 * parameters, and must match in all dimensions except the concatenation
 * axis. The output dimension along @p axis is the sum of the input
 * dimensions along that axis.
 *
 * @param axis       Dimension along which to concatenate (zero-based).
 * @param num_inputs Number of tensors to concatenate.
 * @param inputs     Array of @p num_inputs QHPI_OpRef structures.
 * @return QHPI_OpRef referencing the concatenation result, or an error
 *         reference if validation fails.
 */
QHPI_OpRef qhpi_op_concat(uint32_t axis, uint32_t num_inputs, const QHPI_OpRef *inputs);

/**
 * @brief Callback type: return the required tile shape constraints for an operator.
 *
 * Called at the start of tiling to seed the central tiling system
 * with per-operator constraints. For example, a convolution kernel
 * might require that the depth dimension be 32.
 *
 * Return a shape where each dimension is the required alignment or chunk
 * size. Set a dimension to @ref QHPI_NO_TILING to prevent tiling along
 * that axis.
 *
 * @param op The operator being tiled.
 * @return Required tile shape constraints.
 */
typedef QHPI_Shape (*QHPI_TileShapeRequired)(const QHPI_Op *op);

/**
 * @brief Callback type: legalize a proposed tile shape by adjusting values.
 *
 * Called by the central tiling system with a proposed tile shape
 * based on its heuristics. The callback should adjust dimension sizes
 * as needed to make them legal for this operator (e.g. rounding up to
 * required multiples).
 *
 * @param op    The operator being tiled.
 * @param shape The proposed tile shape to legalize.
 * @return Legalized tile shape with dimensions >= the proposed values.
 */
typedef QHPI_Shape (*QHPI_TileShapeLegalized)(const QHPI_Op *op, const QHPI_Shape *shape);

/**
 * @brief Callback type: build a tiled sub-graph computing a region of the output.
 *
 * Given an output region defined by [@p start, @p start + @p extent), this
 * callback constructs the sub-graph of operators needed to compute that
 * region. It is responsible for determining the corresponding input slices.
 *
 * @param op     The original (untiled) operator.
 * @param start  Per-dimension start offsets of the output tile.
 * @param extent Per-dimension sizes of the output tile.
 * @return Pointer to the root operator of the tiled sub-graph.
 */
typedef const QHPI_Op *(*QHPI_BuildTileOfOp)(const QHPI_Op *op, const QHPI_Shape *start, const QHPI_Shape *extent);

/**
 * @brief Sentinel value returned from QHPI_TileShapeRequired to indicate
 *        that a dimension must not be tiled.
 */
#define QHPI_NO_TILING 0xffffffff

/**
 * @brief Callback type: rewrite an operator into an equivalent sub-graph.
 *
 * Used for both early rewrites (before tiling, when source-level ops are
 * still present) and late rewrites (after tiling, just before layout and
 * placement decisions).
 *
 * @param op The operator to rewrite.
 * @return Pointer to the replacement operator (may be @p op itself if no
 *         rewrite is needed), or an error node.
 */
typedef const QHPI_Op *(*QHPI_RewriteOpFunc)(const QHPI_Op *op);

/**
 * @brief One dimension entry in a chunked memory layout descriptor.
 *
 * Describes how a single dimension is chunked in memory. Entries are
 * ordered from slowest-varying to fastest-varying.
 */
typedef struct {
    uint8_t dim; /**< Dimension index (0-based). */
    uint8_t chunk_size; /**< Chunk size along this dimension. 0 means "entire remaining extent". */
} QHPI_DimInfo;

/**
 * @brief Descriptor for a chunked (blocked) memory layout.
 *
 * Describes how tensor data is organized in memory as a sequence of
 * dimension chunking rules, ordered from slowest to fastest varying.
 *
 * For flat layouts, @c dim_info_size equals @c rank and each entry has
 * @c chunk_size == 0 (meaning the full extent). The default flat
 * layout is: @code {4, 4, {{0,0},{1,0},{2,0},{3,0}}} @endcode
 *
 * For crouton (blocked) layouts, additional entries with non-zero
 * @c chunk_size describe the blocking structure. For example, 8-bit
 * crouton: @code {4, 7, {{0,0},{1,0},{2,0},{3,0}, {1,8},{2,8},{3,32}}} @endcode
 *
 * For indirect (block-table) tensors, entries with non-zero @c chunk_size
 * define individual block dimensions, and entries with @c chunk_size == 0
 * define the block table indexing. Blocks are typically 2 KB.
 */
typedef struct {
    uint32_t rank; /**< Tensor rank. */
    uint32_t dim_info_size; /**< Number of valid entries in @c dim_info. */
    QHPI_DimInfo dim_info[1]; /**< Variable-length array of dimension info (actual size is @c dim_info_size). */
} QHPI_ChunkedMemoryLayout;

/**
 * @brief Standard memory layout identifiers.
 *
 * Each value corresponds to a predefined QHPI_ChunkedMemoryLayout
 * descriptor that can be retrieved with qhpi_chunked_layout().
 * Values >= QHPI_LAYOUT_CUSTOM are defined by plugin usage.
 */
enum {
    QHPI_LAYOUT_ANY = 0, /**< No layout constraint (accepts any layout). */
    QHPI_LAYOUT_SHAPE_ONLY = 1, /**< Shape metadata only, no data storage. */

    QHPI_LAYOUT_FLAT_4 = 2, /**< 4D flat row-major layout. */
    QHPI_LAYOUT_FLAT_5 = 3, /**< 5D flat row-major layout. */
    QHPI_LAYOUT_FLAT_6 = 4, /**< 6D flat row-major layout. */

    QHPI_LAYOUT_SINGULAR = 5, /**< Scalar (single element). */
    QHPI_LAYOUT_NCHW = 6, /**< NCHW channel-first layout. */
    QHPI_LAYOUT_DEPTH_32 = 7, /**< Depth-32 blocked layout. */
    QHPI_LAYOUT_WEIGHTS_8X4 = 8, /**< 8x4 weight packing layout. */

    QHPI_LAYOUT_CROUTON_8 = 9, /**< 8-bit crouton (2 KB blocked) layout. */
    QHPI_LAYOUT_CROUTON_16 = 10, /**< 16-bit crouton layout. */
    QHPI_LAYOUT_CROUTON_32 = 11, /**< 32-bit crouton layout. */
    QHPI_LAYOUT_CROUTON_4X1 = 12, /**< 4x1 crouton variant. */
    QHPI_LAYOUT_CROUTON_2X2 = 13, /**< 2x2 crouton variant. */

    QHPI_LAYOUT_WIDE_CROUTON_8 = 14, /**< 8-bit wide crouton layout. */
    QHPI_LAYOUT_WIDE_CROUTON_32 = 15, /**< 32-bit wide crouton layout. */
    QHPI_LAYOUT_WIDE_CROUTON_2X2 = 16, /**< 2x2 wide crouton variant. */

    QHPI_LAYOUT_CUSTOM = 1000, /**< Base value for plugin-defined custom layouts. */
    QHPI_LAYOUT_INVALID = INT_MAX /**< Sentinel for invalid layout. */
};
typedef uint32_t QHPI_Standard_Layout;

/**
 * @brief Fill in a chunked memory layout descriptor for a standard layout.
 *
 * Writes the QHPI_ChunkedMemoryLayout corresponding to @p standard_layout
 * into the buffer at @p chunked_layout.
 *
 * @param standard_layout The standard layout to look up.
 * @param[out] chunked_layout Buffer to receive the layout descriptor.
 * @param size               Size of the @p chunked_layout buffer in bytes.
 *                           Must be at least large enough for the descriptor
 *                           (use @ref QHPI_MAX_MEMORY_LAYOUT_SIZE for safety).
 * @return @c QHPI_SUCCESS on success.
 * @return @c QHPI_UNSUPPORTED for layouts that have no chunked descriptor
 *         (QHPI_LAYOUT_ANY, QHPI_LAYOUT_SHAPE_ONLY, QHPI_LAYOUT_SINGULAR,
 *         QHPI_LAYOUT_CUSTOM).
 * @return @c QHPI_ERROR_FATAL if @p size is too small.
 */
QHPI_StatusCode qhpi_chunked_layout(QHPI_Standard_Layout standard_layout, QHPI_ChunkedMemoryLayout *chunked_layout,
                                    uint32_t size);

/**
 * @brief Memory location constraint for tensors.
 */
enum {
    QHPI_MEM_LOC_DDR_ONLY = 0, /**< Tensor must reside in DDR. */
    QHPI_MEM_LOC_TCM_ONLY = 1, /**< Tensor must reside in TCM. */
    QHPI_MEM_LOC_DDR_OR_TCM = 2, /**< Tensor may reside in either DDR or TCM. */
    QHPI_MEM_LOC_INVALID = INT_MAX /**< Sentinel for invalid placement. */
};
typedef uint32_t QHPI_MemLoc;

/**
 * @brief Storage access mode for tensor data.
 */
enum {
    QHPI_STORAGE_DIRECT = 0, /**< Data is contiguous; access via raw pointer. */
    QHPI_STORAGE_INDIRECT = 1, /**< Data is in blocks; access via block table. */
    QHPI_STORAGE_DIRECT_OR_INDIRECT = 2 /**< Either storage mode is acceptable. */
};
typedef uint32_t QHPI_Storage;

/**
 * @brief Tensor signature describing the expected properties of one
 *        input or output parameter of a kernel.
 *
 * Used in QHPI_Kernel_v1 to declare what tensor types a kernel accepts.
 */
typedef struct {
    uint32_t element_type; /**< Expected QHPI_Element_Type, or QHPI_ELEMENT_TYPE_ANY. */
    uint32_t layout; /**< Expected QHPI_Standard_Layout, or QHPI_LAYOUT_ANY. */
    uint8_t storage; /**< Expected QHPI_Storage mode. */
    uint8_t mem_placement; /**< Expected QHPI_MemLoc placement. */
} QHPI_Tensor_Signature_v1;

/**
 * @brief Opaque handle to a tensor at execution time.
 *
 * Provides access to tensor data and metadata through the qhpi_tensor_*
 * accessor functions. Only valid during kernel execution callbacks.
 */
typedef struct QHPI_Tensor QHPI_Tensor;

/**
 * @brief Opaque handle to the runtime execution context.
 *
 * Passed to plugin kernel functions at execution time. Provides access
 * to runtime properties such as thread counts, slice information for
 * self-slicing operators, and the synchronization block.
 */
typedef struct QHPI_RuntimeHandle QHPI_RuntimeHandle;

/**
 * @brief Signature for a plugin kernel execution function.
 *
 * Called by the runtime to execute an operator. The same function may be
 * invoked on different resource types (HVX, HMX); the QHPI_RuntimeHandle
 * indicates which resource is allocated for the calling thread.
 *
 * @param handle      Runtime context for this invocation.
 * @param num_outputs Number of output tensors.
 * @param outputs     Array of pointers to writable output tensors.
 * @param num_inputs  Number of input tensors.
 * @param inputs      Array of pointers to read-only input tensors.
 * @return 0 on success, non-zero on error.
 */
typedef uint32_t (*QHPI_Plugin_Function)(QHPI_RuntimeHandle *handle, uint32_t num_outputs, QHPI_Tensor **outputs,
                                         uint32_t num_inputs, const QHPI_Tensor *const *inputs);

/**
 * @brief Signature for a precomputation function.
 *
 * Called once when a graph is loaded, before any inference executions.
 * Allows the kernel to compute data that depends on runtime information
 * (e.g. tensor shapes, quantization) but is constant across inferences.
 *
 * The @p data buffer has size specified by QHPI_Kernel_v1::precomputed_data_size
 * and is zero-initialized before the first call.
 *
 * @param handle      Runtime context.
 * @param data        Writable buffer for precomputed data.
 * @param num_outputs Number of output tensors.
 * @param outputs     Array of pointers to output tensors.
 * @param num_inputs  Number of input tensors.
 * @param inputs      Array of pointers to read-only input tensors.
 * @return 0 on success, non-zero on error.
 */
typedef uint32_t (*QHPI_Precompute_Function)(QHPI_RuntimeHandle *handle, void *data, uint32_t num_outputs,
                                             QHPI_Tensor **outputs, uint32_t num_inputs,
                                             const QHPI_Tensor *const *inputs);

/**
 * @brief Signature for a kernel that uses precomputed data.
 *
 * Called at inference time instead of QHPI_Plugin_Function when
 * precomputed data is available. When this function pointer is set in
 * QHPI_Kernel_v1, the @c function field must be NULL.
 *
 * @param handle           Runtime context.
 * @param precomputed_data Read-only pointer to data populated by the
 *                         QHPI_Precompute_Function.
 * @return 0 on success, non-zero on error.
 */
typedef uint32_t (*QHPI_Plugin_Precomputed_Function)(QHPI_RuntimeHandle *handle, const void *precomputed_data);

/**
 * @brief Signature for an optional kernel matching predicate.
 *
 * When set on a QHPI_Kernel_v1, this function is called during preparation
 * to determine whether the kernel is a valid match for a given operator
 * instance and its input tensors, beyond what the tensor signatures alone
 * can express.
 *
 * @param op         The operator being matched.
 * @param num_inputs Number of input tensors.
 * @param inputs     Array of pointers to read-only input tensors.
 * @return Non-zero if the kernel matches, 0 otherwise.
 */
typedef uint32_t (*QHPI_Kernel_Predicate)(const QHPI_Op *op, const uint32_t num_inputs,
                                          const QHPI_Tensor *const *inputs);

/**
 * @brief Return the logical shape of a tensor (excluding padding).
 * @param tensor Pointer to the tensor to query.
 * @return The tensor's shape.
 */
QHPI_Shape qhpi_tensor_shape(const QHPI_Tensor *tensor);

/**
 * @brief Return the padding applied to a tensor.
 *
 * For indirect crouton (blocked) layouts, returns the per-dimension
 * offset into the first block where valid data begins. For flat
 * layouts, all dimensions are zero.
 *
 * @param tensor Pointer to the tensor to query.
 * @return Per-dimension padding values.
 */
QHPI_Shape qhpi_tensor_padding(const QHPI_Tensor *tensor);

/**
 * @brief Return the padded (allocated) shape of a tensor.
 *
 * This is the logical shape plus any padding required by the memory
 * layout. For flat layouts this equals the logical shape.
 *
 * @param tensor Pointer to the tensor to query.
 * @return The padded shape.
 */
QHPI_Shape qhpi_tensor_padded_shape(const QHPI_Tensor *tensor);

/**
 * @brief Return the element type of a tensor.
 *
 * @param tensor Pointer to the tensor to query.
 * @return The tensor's QHPI_Element_Type.
 */
QHPI_Element_Type qhpi_tensor_type(const QHPI_Tensor *tensor);

/**
 * @brief Return the memory layout of a tensor.
 *
 * @param tensor Pointer to the tensor to query.
 * @return A QHPI_Standard_Layout value identifying the layout.
 */
QHPI_Standard_Layout qhpi_tensor_layout(const QHPI_Tensor *tensor);

/**
 * @brief Return the memory placement of a tensor.
 *
 * @param tensor Pointer to the tensor to query.
 * @return A QHPI_MemLoc value.
 */
QHPI_MemLoc qhpi_tensor_placement(const QHPI_Tensor *tensor);

/**
 * @brief Return the quantization parameters of a tensor.
 *
 * Meaningful only for quantized element types.
 *
 * @param tensor Pointer to the tensor to query.
 * @return The tensor's quantization parameters (zero_offset and stepsize).
 */
QHPI_Quant_Parameters qhpi_tensor_quant_parameters(const QHPI_Tensor *tensor);

/**
 * @brief Return whether a tensor uses indirect (block-table) storage.
 *
 * Distinguishes QHPI_STORAGE_DIRECT from QHPI_STORAGE_INDIRECT at runtime.
 * Prefer this over inferring storage mode from qhpi_tensor_block_table_length()
 * being nonzero: a direct-storage tensor may still report a block table of
 * length 1 (its single contiguous buffer treated as one block), which is not
 * a reliable indirect-storage signal.
 *
 * @param tensor Pointer to the tensor to query.
 * @return Non-zero if the tensor is indirect (block-table) storage, 0 if
 *         direct storage or @p tensor is NULL.
 */
uint32_t qhpi_tensor_is_indirect(const QHPI_Tensor *tensor);

/**
 * @brief Return a pointer to the raw data of a direct-storage tensor.
 *
 * Only valid for tensors with QHPI_STORAGE_DIRECT storage mode.
 * For indirect tensors, use qhpi_tensor_block_table() instead.
 *
 * @param tensor Pointer to the tensor to query.
 * @return Pointer to the contiguous tensor data, or NULL.
 */
void *qhpi_tensor_raw_data(const QHPI_Tensor *tensor);

/**
 * @brief Return the block table of an indirect-storage tensor.
 *
 * For tensors with QHPI_STORAGE_INDIRECT storage mode, returns an array
 * of pointers to individual data blocks (typically 2 KB each).
 *
 * @param tensor Pointer to the tensor to query.
 * @return Pointer to the array of block pointers.
 */
void **qhpi_tensor_block_table(const QHPI_Tensor *tensor);

/**
 * @brief Return the number of blocks in an indirect tensor's block table.
 *
 * @param tensor Pointer to the tensor to query.
 * @return Number of entries in the block table.
 */
uint32_t qhpi_tensor_block_table_length(const QHPI_Tensor *tensor);

/**
 * @brief Return the shape of individual blocks in a blocked tensor.
 *
 * For crouton and other blocked layouts, returns the per-dimension size
 * of each block. For non-blocked layouts, returns zeros.
 *
 * @param tensor Pointer to the tensor to query.
 * @return The block shape.
 */
QHPI_Shape qhpi_tensor_block_shape(const QHPI_Tensor *tensor);

/**
 * @brief Signature for a cost estimation function.
 *
 * Returns an approximate cycle count for executing the operator with the
 * given inputs. Used by the scheduler for ordering decisions; not used
 * to distinguish between matching kernel instances.
 *
 * @param handle     Runtime context.
 * @param num_inputs Number of input tensors.
 * @param inputs     Array of pointers to read-only input tensors.
 * @return Estimated execution cost in cycles.
 */
typedef float (*QHPI_Cost_Function)(QHPI_RuntimeHandle *handle, const uint32_t num_inputs,
                                    const QHPI_Tensor *const *inputs);

/**
 * @brief Read a tensor element in logical coordinates as float.
 *
 * Works for any layout (Flat, Crouton, WideCrouton, etc.) and any
 * quantized or floating element type.
 *
 * @param tensor Tensor handle.
 * @param rank   Rank of coordinate list.
 * @param ...    rank logical indices.
 *
 * @return Logical tensor value converted to float.
 */
float qhpi_tensor_get(const QHPI_Tensor *tensor, uint32_t rank, ...);

/**
 * @brief Write a tensor element in logical coordinates from float.
 *
 * Works for any layout and quantization.
 */
void qhpi_tensor_set(const QHPI_Tensor *tensor, uint32_t rank, ...);

/**
 * @brief Description of a single kernel implementation for an operator.
 *
 * A kernel is one concrete implementation of an operator, targeting
 * specific tensor types, layouts, and hardware resources. An operator
 * may have multiple kernels; the compiler selects the best match based
 * on tensor signatures and the optional predicate.
 *
 */
typedef struct {
    const char *function_name; /**< Name of the kernel entry point. */
    QHPI_Plugin_Function function; /**< Execution function pointer, or NULL if using precomputed path. */
    uint8_t resources; /**< Bitmask of QHPI_Resource flags indicating required hardware. */

    uint8_t source_destructive : 1; /**< Non-zero if the first input and first output may alias. */

    uint8_t multithreaded : 1; /**< Non-zero if the kernel supports self-slicing across multiple threads. */

    uint8_t variable_inputs : 1; /**< Non-zero if the kernel accepts more than @c min_inputs inputs. */
    uint8_t variable_outputs : 1; /**< Non-zero if the kernel accepts more than @c min_outputs outputs. */

    uint32_t min_inputs; /**< Minimum number of inputs required. */
    QHPI_Tensor_Signature_v1 *input_signature; /**< Array of @c min_inputs tensor signatures for inputs.
                                                    When @c variable_inputs is set, the last signature
                                                    applies to all additional inputs. */
    uint32_t min_outputs; /**< Minimum number of outputs required. */
    QHPI_Tensor_Signature_v1 *output_signature; /**< Array of @c min_outputs tensor signatures for outputs.
                                                     When @c variable_outputs is set, the last signature
                                                     applies to all additional outputs. */

    QHPI_Cost_Function cost_function; /**< Optional cost estimator for scheduling (may be NULL). */

    uint32_t sync_block_size; /**< Size in bytes of shared zero-initialized synchronization memory
                                   accessible via qhpi_sync_block(). 0 if not needed. */

    uint32_t precomputed_data_size; /**< Size in bytes of the precomputed data buffer. 0 if not used. */

    QHPI_Precompute_Function do_precomputation_function; /**< Called at graph load to populate precomputed data.
                                                              NULL if precomputation is not used. */

    QHPI_Plugin_Precomputed_Function function_with_precomputed_data; /**< Execution function using precomputed data.
                                                                         When set, @c function must be NULL. */

    QHPI_Kernel_Predicate predicate; /**< Optional predicate for additional matching logic (may be NULL). */

    void *Reserved_1; /**< Reserved for future use. Must be set to NULL. */
    void *Reserved_2; /**< Reserved for future use. Must be set to NULL. */
    void *Reserved_3; /**< Reserved for future use. Must be set to NULL. */
    void *Reserved_4; /**< Reserved for future use. Must be set to NULL. */
} QHPI_Kernel_v1;

/**
 * @brief Hardware resource flags for kernel requirements.
 *
 * Specified as a bitmask in QHPI_Kernel_v1::resources. When the kernel
 * is not self-sliced, exactly one thread of the specified type calls it.
 * When self-sliced (multithreaded), multiple threads of the specified
 * type call the kernel; use qhpi_num_slices() and qhpi_slice_number()
 * to coordinate.
 */
enum {
    QHPI_RESOURCE_MAIN = 1, /**< Requires a main (scalar) thread. */
    QHPI_RESOURCE_HVX = 2, /**< Requires an HVX (vector) thread. */
    QHPI_RESOURCE_HMX = 4, /**< Requires an HMX (matrix) thread. */
    QHPI_RESOURCE_EXCLUSIVE = 0xf, /**< Requires all threads. */
};
typedef uint32_t QHPI_Resource;

/**
 * @brief Return the total number of slices for a self-slicing operator.
 *
 * For multithreaded kernels, the runtime calls the kernel function
 * @c num_slices times, each with a different slice index.
 *
 * @param handle Runtime context.
 * @return Total number of slices.
 */
uint32_t qhpi_num_slices(QHPI_RuntimeHandle *handle);

/**
 * @brief Return the slice index for the current invocation.
 *
 * For self-slicing operators, each thread receives a unique slice index
 * in the range [0, qhpi_num_slices()).
 *
 * @param handle Runtime context.
 * @return Current slice index.
 */
uint32_t qhpi_slice_number(QHPI_RuntimeHandle *handle);

/**
 * @brief Return a pointer to the shared synchronization block.
 *
 * The sync block is a region of zero-initialized memory whose size is
 * specified by QHPI_Kernel_v1::sync_block_size. It is shared and
 * writable by all threads in a self-sliced computation, enabling
 * inter-thread coordination.
 *
 * @param handle Runtime context.
 * @return Pointer to the sync block, or NULL if @c sync_block_size is 0.
 */
void *qhpi_sync_block(QHPI_RuntimeHandle *handle);

/**
 * @brief Return the resource type of the thread currently executing the kernel.
 *
 * Useful for ops registered with @c QHPI_RESOURCE_EXCLUSIVE, which dispatch
 * on multiple thread types simultaneously and may want to specialize their
 * work per thread.
 *
 * @param handle Runtime context.
 * @return 0 for the main thread, otherwise one of @c QHPI_RESOURCE_HVX,
 *         @c QHPI_RESOURCE_HMX, or @c QHPI_RESOURCE_HLX.
 */
uint32_t qhpi_thread_resources(QHPI_RuntimeHandle *handle);

/**
 * @brief Return the SRM compute resource context ID held by the graph.
 *
 * @param handle Runtime context.
 * @return SRM context ID, or 0 if not available.
 */
uint32_t qhpi_srm_context(QHPI_RuntimeHandle *handle);

/**
 * @brief Description of a plugin operator, including all kernel
 *        implementations and optimization callbacks.
 *
 * One QHPI_OpInfo_v1 is registered per operator name within a package.
 * If an operator is used only transiently during graph rewriting, it is
 * not necessary to provide kernel implementations.
 *
 * The following fields are only used while preparing the graph and may be
 * NULL/0 when defining execute-only operators:
 *
 * - @c early_rewrite
 * - @c shape_required
 * - @c shape_legalized
 * - @c tile_output
 * - @c build_tile
 * - @c late_rewrite
 */
typedef struct {
    const char *name; /**< Operator name including package prefix. */
    uint32_t num_kernels; /**< Number of kernel implementations. */
    QHPI_Kernel_v1 *kernels; /**< Array of @c num_kernels kernel descriptors. */
    QHPI_RewriteOpFunc early_rewrite; /**< Rewrite callback invoked before tiling (source-level ops available). */
    QHPI_TileShapeRequired shape_required; /**< Returns required tile shape constraints. */
    QHPI_TileShapeLegalized shape_legalized; /**< Legalizes a proposed tile shape. */
    unsigned tile_output; /**< Index of the output to tile. */
    QHPI_BuildTileOfOp build_tile; /**< Builds the tiled sub-graph for an output region. */
    QHPI_RewriteOpFunc late_rewrite; /**< Rewrite callback invoked after tiling. */
    void *Reserved_1; /**< Reserved for future use. Must be set to NULL. */
    void *Reserved_2; /**< Reserved for future use. Must be set to NULL. */
    void *Reserved_3; /**< Reserved for future use. Must be set to NULL. */
    void *Reserved_4; /**< Reserved for future use. Must be set to NULL. */
} QHPI_OpInfo_v1;

/**
 * @brief Test whether an operator is an error marker.
 *
 * Error markers have 0 inputs, 0 outputs, and the name "$Error".
 *
 * @param op Pointer to the operator to test.
 * @return Non-zero if @p op is an error marker, 0 otherwise.
 */
uint32_t qhpi_op_is_error(const QHPI_Op *op);

/**
 * @brief Return the descriptive text of an error marker operator.
 * @param op Pointer to an error marker operator.
 * @return Null-terminated error description string.
 */
const char *qhpi_op_error_description(const QHPI_Op *op);

/**
 * @brief Test whether an output definition represents an error.
 *
 * An error output has element type QHPI_ELEMENT_TYPE_INVALID.
 *
 * @param output Pointer to the output definition to test.
 * @return Non-zero if the output is an error, 0 otherwise.
 */
uint32_t qhpi_output_is_error(const QHPI_OutputDef *output);

/**
 * @brief Return the descriptive text for an error output.
 *
 * @param root   An operator providing the graph context.
 * @param output Pointer to an error output definition.
 * @return Null-terminated error description string, or
 *         "invalid error reference" if the output is not a valid error.
 */
const char *qhpi_output_error_description(const QHPI_Op *root, const QHPI_OutputDef *output);

/**
 * @brief Create an error marker operator with a description.
 *
 * Error nodes propagate through the graph and can be detected with
 * qhpi_op_is_error(). Use this to signal errors during rewrite or
 * tiling callbacks.
 *
 * @param root        An existing operator used to identify the graph context.
 * @param description Null-terminated error description string.
 * @return Pointer to the new error marker operator.
 */
const QHPI_Op *qhpi_op_create_error(const QHPI_Op *root, const char *description);

/**
 * @brief Register a collection of plugin operators.
 *
 * This is the main entry point for making custom operators available
 * to the compiler. All operators in the array are registered under
 * the given package name.
 *
 * @param num_ops   Number of operators in the @p operators array.
 * @param operators Array of @p num_ops QHPI_OpInfo_v1 structures.
 * @param package   Null-terminated package name used as a namespace prefix.
 * @return @c QHPI_SUCCESS on success, or an error status code.
 */
QHPI_StatusCode qhpi_register_ops_v1(uint32_t num_ops, QHPI_OpInfo_v1 *operators, const char *package);

#ifdef __cplusplus
}
#endif

#endif // #ifndef QHPI_QHPI
