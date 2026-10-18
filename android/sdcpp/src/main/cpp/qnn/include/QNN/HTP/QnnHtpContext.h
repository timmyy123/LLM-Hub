//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All rights reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.s
//
//==============================================================================

/**
 *  @file
 *  @brief QNN HTP component Context API.
 *
 *         The interfaces in this file work with the top level QNN
 *         API and supplements QnnContext.h for HTP backend
 */

#ifndef QNN_HTP_CONTEXT_H
#define QNN_HTP_CONTEXT_H

#include "QnnContext.h"

#ifdef __cplusplus
extern "C" {
#endif

//=============================================================================
// Macros
//=============================================================================

//=============================================================================
// Data Types
//=============================================================================

/**
 * @brief This enum provides different HTP context configuration
 *        options associated with QnnContext
 */
typedef enum {
  QNN_HTP_CONTEXT_CONFIG_OPTION_WEIGHT_SHARING_ENABLED                  = 1,
  QNN_HTP_CONTEXT_CONFIG_OPTION_REGISTER_MULTI_CONTEXTS                 = 2,
  QNN_HTP_CONTEXT_CONFIG_OPTION_FILE_READ_MEMORY_BUDGET                 = 3,
  QNN_HTP_CONTEXT_CONFIG_OPTION_DSP_MEMORY_PROFILING_ENABLED            = 4,
  QNN_HTP_CONTEXT_CONFIG_OPTION_SHARE_RESOURCES                         = 5,
  QNN_HTP_CONTEXT_CONFIG_OPTION_IO_MEM_ESTIMATION                       = 6,
  QNN_HTP_CONTEXT_CONFIG_OPTION_PREPARE_ONLY                            = 7,
  QNN_HTP_CONTEXT_CONFIG_OPTION_INIT_ACCELERATION                       = 8,
  QNN_HTP_CONTEXT_CONFIG_OPTION_SKIP_VALIDATION_ON_BINARY_SECTION       = 9,
  QNN_HTP_CONTEXT_CONFIG_OPTION_SHARE_RESOURCES_OPTIMIZATION_TYPE       = 10,
  QNN_HTP_CONTEXT_CONFIG_OPTION_USE_EXTENDED_UDMA                       = 11,
  QNN_HTP_CONTEXT_CONFIG_OPTION_REGISTER_CONCURRENT_RESOURCE_SHARING    = 12,
  QNN_HTP_CONTEXT_CONFIG_OPTION_LORA_WEIGHT_SHARING_ENABLED             = 13,
  QNN_HTP_CONTEXT_CONFIG_OPTION_RESERVED_14                             = 14,
  QNN_HTP_CONTEXT_CONFIG_OPTION_RESERVED_15                             = 15,
  QNN_HTP_CONTEXT_CONFIG_OPTION_LORA_WEIGHT_SHARING_RAM_PRELOAD_ENABLED = 16,
  QNN_HTP_CONTEXT_CONFIG_OPTION_REUSED_IO_LIMIT                         = 17,
  QNN_HTP_CONTEXT_CONFIG_OPTION_REFERENCE_WEIGHT_SHARING_ENABLED        = 18,
  QNN_HTP_CONTEXT_CONFIG_OPTION_CONCURRENT_DESERIALIZATION_EXTRA_SIZE   = 19,
  QNN_HTP_CONTEXT_CONFIG_OPTION_GRAPH_SPLITTING_ENABLED                 = 20,  // deprecated
  QNN_HTP_CONTEXT_CONFIG_OPTION_CONCURRENT_DESERIALIZATION_PATCH        = 21,
  QNN_HTP_CONTEXT_CONFIG_OPTION_GRAPH_SPLITTING_CONFIGS                 = 22,
  QNN_HTP_CONTEXT_CONFIG_OPTION_LORA_PRELOAD_BUFFER_GROUPS              = 23,
  QNN_HTP_CONTEXT_CONFIG_OPTION_RESERVED_24                             = 24,
  QNN_HTP_CONTEXT_CONFIG_OPTION_PARALLEL_WEIGHT_LOADING                 = 25,
  QNN_HTP_CONTEXT_CONFIG_OPTION_UNKNOWN                                 = 0x7fffffff
} QnnHtpContext_ConfigOption_t;

typedef struct {
  // Handle referring to the first context associated to a group. When a new
  // group is to be registered, the following value must be 0.
  Qnn_ContextHandle_t firstGroupHandle;
  // Max spill-fill buffer to be allocated for the group of context in bytes.
  // The value that is passed during the registration of the first context to
  // a group is taken. Subsequent configuration of this value is disregarded.
  uint64_t maxSpillFillBuffer;
} QnnHtpContext_GroupRegistration_t;

// This enum is supported only with the QnnContext_createFromBinaryListAsync API, when
// shareResources is true; otherwise, it is ignored. This enumeration allows users to specify how
// graphs are going to be executed, providing QNN with hints for optimizing memory.
typedef enum {
  // Default value if no user input is provided.
  // This type is used for sequential graph execution, optimizing both VA and memory.
  SEQUENTIAL_WITH_VA_OPTIMIZATION,
  // This type is used for sequential graph execution, optimizing memory.
  SEQUENTIAL_WITHOUT_VA_OPTIMIZATION,
  // This type is used for concurrent resource sharing, optimizing memory by sharing
  // resources across contexts with the same priority level.
  CONCURRENT_OPTIMIZATION,
} QnnHtpContext_ShareResourcesOptimizationType_t;

typedef struct {
  // When enabled, the graph will be split based on its structure and
  // each part compiled as an independent subgraph.
  bool graphSplittingEnabled;
} QnnHtpContext_GraphSplit_t;

// Intent for the parallel weight-copy thread during context creation from binary.
// Used with QNN_HTP_CONTEXT_CONFIG_OPTION_PARALLEL_WEIGHT_LOADING.
typedef enum {
  // Default. Size-threshold heuristic decides on the legacy path; callback path stays non-parallel.
  QNN_HTP_PARALLEL_WEIGHT_LOADING_MODE_ADAPTIVE = 0,
  // Unconditionally enable, regardless of weights size.
  QNN_HTP_PARALLEL_WEIGHT_LOADING_MODE_ON = 1,
  // Unconditionally disable.
  QNN_HTP_PARALLEL_WEIGHT_LOADING_MODE_OFF = 2,
} QnnHtpContext_ParallelWeightLoadingMode_t;

// Parallel weight-loading configuration for QnnContext_createFromBinary.
// Used with QNN_HTP_CONTEXT_CONFIG_OPTION_PARALLEL_WEIGHT_LOADING.
typedef struct {
  // Intent for the parallel weight-copy thread. When left at ADAPTIVE (0, the default),
  // the size-threshold heuristic below decides on the legacy path; the callback path stays
  // non-parallel. See QnnHtpContext_ParallelWeightLoadingMode_t for ON / OFF semantics.
  QnnHtpContext_ParallelWeightLoadingMode_t mode;
  // Optional override for QNN's internal shared-weights size threshold (in MB) used by
  // ADAPTIVE on the legacy path.
  //   - 0 (default): no override — QNN uses its internal default.
  //   - Non-zero:    threshold override in MB; shared-weights blobs at or above this size
  //                  are copied in parallel; smaller blobs are not.
  // Only consulted when mode == ADAPTIVE. To force parallel loading regardless of size,
  // use mode = ON instead.
  uint64_t thresholdInMbOverride;
} QnnHtpContext_ParallelWeightLoading_t;

/**
 * Describes a single group of graph names that share one LoRA preload buffer.
 * graphNames is a NULL-terminated array of C-string graph names.
 */
typedef struct {
  const char** graphNames;
  uint32_t numGraphs;
} QnnHtpContext_LoraPreloadGroup_t;

/**
 * Configures LoRA RAM preload buffer groups.
 * groups is an array of QnnHtpContext_LoraPreloadGroup_t, each describing
 * a set of graphs that share one preload buffer.
 * numGroups is the number of entries in groups.
 * Graphs not listed in any group each receive their own dedicated buffer.
 */
typedef struct {
  const QnnHtpContext_LoraPreloadGroup_t* groups;
  uint32_t numGroups;
} QnnHtpContext_LoraPreloadBufferGroups_t;

//=============================================================================
// Public Functions
//=============================================================================

//------------------------------------------------------------------------------
//   Implementation Definition
//------------------------------------------------------------------------------

// clang-format off

/**
 * @brief        Structure describing the set of configurations supported by context.
 *               Objects of this type are to be referenced through QnnContext_CustomConfig_t.
 *
 *               The struct has two fields - option and a union of config values
 *               Based on the option corresponding item in the union can be used to specify
 *               config.
 *
 *               Below is the Map between QnnHtpContext_CustomConfig_t and config value
 *
 *               \verbatim embed:rst:leading-asterisk
 *               +----+----------------------------------------------------------------------+--------------------------------------------------+
 *               | #  | Config Option                                                        | Configuration Struct/value                       |
 *               +====+======================================================================+==================================================+
 *               | 1  | QNN_HTP_CONTEXT_CONFIG_OPTION_WEIGHT_SHARING_ENABLED                 | bool                                             |
 *               +====+======================================================================+==================================================+
 *               | 2  | QNN_HTP_CONTEXT_CONFIG_OPTION_REGISTER_MULTI_CONTEXTS                | QnnHtpContext_GroupRegistration_t                |
 *               +====+======================================================================+==================================================+
 *               | 3  | QNN_HTP_CONTEXT_CONFIG_OPTION_FILE_READ_MEMORY_BUDGET                | uint64_t                                         |
 *               +====+======================================================================+==================================================+
 *               | 4  | QNN_HTP_CONTEXT_CONFIG_OPTION_DSP_MEMORY_PROFILING_ENABLED           | bool                                             |
 *               +====+======================================================================+==================================================+
 *               | 5  | QNN_HTP_CONTEXT_CONFIG_OPTION_SHARE_RESOURCES                        | bool                                             |
 *               +====+======================================================================+==================================================+
 *               | 6  | QNN_HTP_CONTEXT_CONFIG_OPTION_IO_MEM_ESTIMATION                      | bool                                             |
 *               +====+======================================================================+==================================================+
 *               | 7  | QNN_HTP_CONTEXT_CONFIG_OPTION_PREPARE_ONLY                           | bool                                             |
 *               +====+======================================================================+==================================================+
 *               | 8  | QNN_HTP_CONTEXT_CONFIG_OPTION_INIT_ACCELERATION                      | bool                                             |
 *               +====+======================================================================+==================================================+
 *               | 9  | QNN_HTP_CONTEXT_CONFIG_OPTION_SKIP_VALIDATION_ON_BINARY_SECTION      | bool                                             |
 *               +====+======================================================================+==================================================+
 *               | 10 | QNN_HTP_CONTEXT_CONFIG_OPTION_SHARE_RESOURCES_OPTIMIZATION_TYPE      | QnnHtpContext_ShareResourcesOptimizationType_t   |
 *               +====+======================================================================+==================================================+
 *               | 11 | QNN_HTP_CONTEXT_CONFIG_OPTION_USE_EXTENDED_UDMA                      | bool                                             |
 *               +====+======================================================================+==================================================+
 *               | 12 | QNN_HTP_CONTEXT_CONFIG_OPTION_REGISTER_CONCURRENT_RESOURCE_SHARING   | QnnHtpContext_GroupRegistration_t                |
 *               +====+======================================================================+==================================================+
 *               | 13 | QNN_HTP_CONTEXT_CONFIG_OPTION_LORA_WEIGHT_SHARING_ENABLED            | bool                                             |
 *               +====+======================================================================+==================================================+
 *               | 16 | QNN_HTP_CONTEXT_CONFIG_OPTION_LORA_WEIGHT_SHARING_RAM_PRELOAD_ENABLED| bool                                             |
 *               +====+======================================================================+==================================================+
 *               | 17 | QNN_HTP_CONTEXT_CONFIG_OPTION_REUSED_IO_LIMIT                        | uint64_t                                         |
 *               +====+======================================================================+==================================================+
 *               | 18 | QNN_HTP_CONTEXT_CONFIG_OPTION_REFERENCE_WEIGHT_SHARING_ENABLED       | bool                                             |
 *               +====+======================================================================+==================================================+
 *               | 19 | QNN_HTP_CONTEXT_CONFIG_OPTION_CONCURRENT_DESERIALIZATION_EXTRA_SIZE  | uint32_t                                         |
 *               +====+======================================================================+==================================================+
 *               | 20 | QNN_HTP_CONTEXT_CONFIG_OPTION_GRAPH_SPLITTING_ENABLED                | bool                                             |
 *               +====+======================================================================+==================================================+
 *               | 21 | QNN_HTP_CONTEXT_CONFIG_OPTION_CONCURRENT_DESERIALIZATION_PATCH       | int                                              |
 *               +====+======================================================================+==================================================+
 *               | 22 | QNN_HTP_CONTEXT_CONFIG_OPTION_GRAPH_SPLITTING_CONFIGS                | QnnHtpContext_GraphSplit_t                       |
 *               +====+======================================================================+==================================================+
 *               | 23 | QNN_HTP_CONTEXT_CONFIG_OPTION_LORA_PRELOAD_BUFFER_GROUPS             | QnnHtpContext_LoraPreloadBufferGroups_t           |
 *               +====+======================================================================+==================================================+
 *               | 25 | QNN_HTP_CONTEXT_CONFIG_OPTION_PARALLEL_WEIGHT_LOADING                | QnnHtpContext_ParallelWeightLoading_t             |
 *               +----+----------------------------------------------------------------------+--------------------------------------------------+
 *               \endverbatim
 */
typedef struct QnnHtpContext_CustomConfig {
  QnnHtpContext_ConfigOption_t option;
  union UNNAMED {
    // This field sets the weight sharing which is by default false
    bool weightSharingEnabled;
    QnnHtpContext_GroupRegistration_t groupRegistration;
    // - Init time may be impacted depending the value set below
    // - Value should be grather than 0 and less than or equal to the file size
    //    - If set to 0, the feature is not utilized
    //    - If set to greater than file size, min(fileSize, fileReadMemoryBudgetInMb) is used
    // - As an example, if value 2 is passed, it would translate to (2 * 1024 * 1024) bytes
    uint64_t fileReadMemoryBudgetInMb;
    bool dspMemoryProfilingEnabled;
    // This field enables resource optimization. When it is set to true optimizations are
    // done based on QnnHtpContext_ShareResourcesOptimizationType_t setting.
    // Note This configuration option is only supported when using QnnContext_createFromBinaryListAsync API.
    bool shareResources;
    // This field enables I/O memory estimation during QnnContext_createFromBinary API when multiple
    // PDs are available. When enabled, it estimates the total size of the I/O tensors required by
    // the context to ensure sufficient space on the PD before deserialization. This feature helps
    // with memory registration failures in large models.
    // Note that enabling this feature increases peak RAM usage during context initialization phase
    // in QnnContext_createFromBinary, but sustained RAM remains unaffected.
    bool ioMemEstimation;
    // This field enables model preparation without mapping its content on the DSP side. It is
    // useful when a model needs to be prepared on the device but executed through a serialized
    // binary method. This prevents extra mapping onto the DSP VA space. Set this flag only when
    // creating the context.
    bool isPrepareOnly;
    // This field enables initialization acceleration, which is disabled by default.
    // If set to true, the DSP will utilize all hardware threads to accelerate deserialization.
    // It is not recommended to execute graphs simultaneously, as this will significantly degrade
    // performance.
    // Note that this feature may not be effective for small graphs with a few number of ops.
    bool initAcceleration;
    // This field enables crc32 check skip in Lora super adapter apply, which is disabled by default.
    // If set to true, crc32 check for non-base adapter in super adapter apply use case will be
    // skipped to improve time cost.
    // Note that base adapter in super adaper never do crc32 check, therefore, their apply time cost
    // won't improve by turning this config option on.
    bool skipValidationOnBinarySection;
    // If shareResources is true:
    // shareResOptType is read. If no value is set by the user,
    // the default value of QnnHtpContext_ShareResourcesOptimizationType_t is used.
    // If shareResources is false:
    // shareResOptType is ignored.
    // Note: This configuration option is only supported when using the QnnContext_createFromBinaryListAsync API.
    QnnHtpContext_ShareResourcesOptimizationType_t shareResOptType;
    // This field enables preparing graphs, associated with this context, with far-mapping enabled so that weights
    // and spill/fill buffer are mapped to the far region of the DSP which is helpful if PD's limited VA space is
    // exhausted. Total RAM usage may increase if used together with shared weights. Only available for Hexagon
    // arch v81 and above.
    bool useExtendedUdma;
    // This field enables concurrent resource sharing among graphs with the same priority level
    // during the QnnContext_createFromBinary API on devices that support this capability.
    QnnHtpContext_GroupRegistration_t concurrentGroupRegistration;
    // This field sets the lora weight sharing. When it is set to true, one additional replaceable weight blob
    // that contains the RP shared by all graphs will be generated and maintained. It is disabled by default.
    bool loraWeightSharingEnabled;
    // This field is an additional option for loraWeightSharingEnabled. When set to true, it enables
    // RAM Preload operations where each graph maintains its own replaceable weight blob instead of
    // using a shared blob. It is disabled by default.
    bool loraWeightSharingRamPreloadEnabled;
    // This field configures LoRA RAM preload buffer groups. Effective only when
    // loraWeightSharingRamPreloadEnabled is true.
    // Each inner array is a group of graph names that share one preload buffer.
    // Graphs not listed in any group each get their own dedicated buffer.
    // Example: groups = [["g1","g4"],["g5","g6","g2","g3"]] means g1 and g4
    // share buffer 0, while g5/g6/g2/g3 share buffer 1.
    const QnnHtpContext_LoraPreloadBufferGroups_t* loraPreloadBufferGroups;
    // This field optimizes IO memory management during QnnContext_createFromBinary and
    // QnnContext_createFromBinaryListAsync APIs. For createFromBinary, it is a per-context
    // property. For createFromBinaryListAsync, it is a group-level property passed via
    // listConfig, where all contexts in the group share a single IO buffer of this size.
    // Unit is in MB. As an example, if 2 is passed in, it is equivalent to (2 * 1024 * 1024) bytes.
    // User shall not map more than this limit and DSP will have no guarantee if user map more
    // than this limit indicates.
    uint64_t reusedIoLimitMb;
    // This field enables weight sharing between context binaries targeting different SoCs.
    // The first context's weights become a reference shared with subsequent context binaries.
    // This is only supported via the DLC workflow.
    // Limitation: Weight sharing performance may be degraded when reference weight sharing is
    // enabled across uDMA and non-uDMA context binaries, or across single-core and multi-core
    // context binaries. Total RAM usage may increase if weight sharing is not optimal.
    // This flag is disabled by default.
    bool referenceWeightSharingEnabled;
    // This field is to store the extra size in KB for backward compatible concurrent deserialization.
    // Unit is in KB. Maximum value is 102400 KB (100 MB).
    // If the value exceeds the maximum, it will be clamped to 102400 KB.
    uint32_t concurrentDeserializationExtraSizeKB;
    // @deprecated Use graphSplittingConfigs instead.
    // This field sets the graph splitting option which is default set to false.
    // When enabled, it will split the graph based on its structure and
    // compile each part as an independent subgraph.
    bool graphSplittingEnabled;
    // This field is to store the file descriptor to the patch file for backward compatible concurrent deserialization.
    int concurrentDeserializationPatchFd;
    // This field sets the graph splitting configurations.
    // Use this in preference to the deprecated graphSplittingEnabled bool above.
    QnnHtpContext_GraphSplit_t graphSplittingConfigs;
    // Parallel weight-loading configuration for QnnContext_createFromBinary.
    // Fields defaulted to zero (mode=ADAPTIVE, thresholdInMbOverride=0) preserve today's behavior.
    // See QnnHtpContext_ParallelWeightLoading_t for full semantics.
    QnnHtpContext_ParallelWeightLoading_t parallelWeightLoading;
  };
} QnnHtpContext_CustomConfig_t;

/// QnnHtpContext_CustomConfig_t initializer macro
#define QNN_HTP_CONTEXT_CUSTOM_CONFIG_INIT            \
  {                                                   \
    QNN_HTP_CONTEXT_CONFIG_OPTION_UNKNOWN, /*option*/ \
    {                                                 \
      false                          /*weightsharing*/\
    }                                                 \
  }

/**
 * @brief Structure describing the set of properties supported by context.
 *        Objects of this type are to be referenced through QnnContext_CustomProperty_t.
 *        Used by QnnContext_getProperty.
 */
typedef enum {
  // get the alignment requirement of persistent buffers
  QNN_HTP_CONTEXT_GET_PROP_BUFFER_START_ALIGNMENT = 1,
  // get the size requirement of spill/fill buffer
  QNN_HTP_CONTEXT_GET_PROP_MAX_SPILLFILL_BUFFER_SIZE = 2,
  // get the size requirement of persistent weights buffer
  QNN_HTP_CONTEXT_GET_PROP_WEIGHTS_BUFFER_SIZE = 3,
  QNN_HTP_CONTEXT_GET_PROP_RESERVED_4 = 4,
  QNN_HTP_CONTEXT_GET_PROP_RESERVED_5 = 5,
  // get the size requirement of VTCM Backup buffer
  QNN_HTP_CONTEXT_GET_PROP_MAX_VTCMBACKUP_BUFFER_SIZE = 6,
  QNN_HTP_CONTEXT_GET_PROP_RESERVED_7 = 7,
  // Unused, present to ensure 32 bits.
  QNN_HTP_CONTEXT_GET_PROP_UNDEFINED = 0x7fffffff
} QnnHtpContext_GetPropertyOption_t;

// used by QnnContext_getProperty
typedef struct {
  QnnHtpContext_GetPropertyOption_t option;
  union UNNAMED {
    uint64_t bufferStartAlignment;
    uint64_t spillfillBufferSize;
    uint64_t weightsBufferSize;
    uint64_t vtcmBackupBufferSize;
  };
} QnnHtpContext_CustomProperty_t;

// clang-format off
/// QnnHtpContext_CustomProperty_t initializer macro
#define QNN_HTP_CONTEXT_CUSTOM_PROPERTY_INIT                         \
  {                                                                  \
    QNN_HTP_CONTEXT_GET_PROP_UNDEFINED,                 /*option*/   \
    0                                         /*scratchBufferSize*/  \
  }
// clang-format on

// clang-format on
#ifdef __cplusplus
}  // extern "C"
#endif

#endif
