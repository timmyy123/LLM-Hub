//=============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================

/**
 *  @file
 *  @brief QNN Graph Transformer component Graph API.
 *
 *         The interfaces in this file work with the top level QNN
 *         API and supplements graphs to enable graph transformation. This is a
 *         low-level API that allows users to configure graph transformations at
 *         the pass level.
 */

#ifndef QNN_GRAPH_TRANSFORMER_CONFIG_H
#define QNN_GRAPH_TRANSFORMER_CONFIG_H

#include "QnnGraph.h"

#ifdef __cplusplus
extern "C" {
#endif


/**
 * @brief A struct for passing pass names to the QNN Graph Transformer.
 *        This structure encapsulates an array of pass names and its size
 *        for configuring which optimization passes to enable or disable.
 *
 *        Usage example:
 *        @code
 *        char* passNames[] = {"pass1", "pass2", "pass3"};
 *        QnnGraphTransformer_PassNames_t passes = {passNames, 3};
 *        @endcode
 */
typedef struct {
  // Array of pass name strings. Each string represents the name of an optimization pass.
  // The caller is responsible for ensuring the strings remain valid during usage.
  char **data;
  // Number of pass names in the data array. Must be >= 0.
  int size;
} QnnGraphTransformer_PassNames_t;

/**
 * @brief This enum provides different QNN Graph Transformer configuration
 *        options that can be used to control graph transformation behavior.
 *
 *        These options allow fine-grained control over the QNN Graph Transformer
 *        and its individual optimization passes.
 */
typedef enum {
  // Reserve a dedicated high-range block (0x00100000-0x001FFFFF) for QGT custom
  // config options.

  // Enable or disable the QNN Graph Transformer entirely.
  // When enabled, the optimizer will run with default pass configuration.
  QNN_GRAPH_TRANSFORMER_CUSTOM_CONFIG_OPTION_ENABLE_QNN_GRAPH_TRANSFORMS    = 0x00100001,
  // Specify which optimization passes to explicitly enable.
  // This allows selective enabling of specific optimization passes.
  QNN_GRAPH_TRANSFORMER_CUSTOM_CONFIG_OPTION_ENABLE_PASSES                  = 0x00100002,
  // Specify which optimization passes to explicitly disable.
  // This allows selective disabling of specific optimization passes.
  QNN_GRAPH_TRANSFORMER_CUSTOM_CONFIG_OPTION_DISABLE_PASSES                 = 0x00100003,
  // Dump DLC before and after applying optimizations.
  QNN_GRAPH_TRANSFORMER_CUSTOM_CONFIG_OPTION_DUMP_DLC                       = 0x00100004,
  // Unused, present to ensure 32 bits.
  QNN_GRAPH_TRANSFORMER_CUSTOM_CONFIG_OPTION_UNKNOWN                        = 0x7fffffff
} QnnGraphTransformer_CustomConfigOption_t;

/**
 * @brief An enum to specify the target backend for graph transformation.
 *        This allows the graph transformer to apply backend-specific optimizations
 *        and configurations during the transformation process.
 */
typedef enum {
  // Apply transformations specifically optimized for the HTP backend.
  // This enables HTP-specific optimizations and configurations during graph transformation.
  // Mapping is added only for the backends which are added in BackendAwareness module
  HTP  = 1,
  LPAI = 2,
  CPU  = 3,
  GPU  = 4,
  DSP  = 5,
  HTA  = 6,
  // Unused, present to ensure 32 bits.
  UNKNOWN = 0x7fffffff
} QnnGraphTransformer_BackendType_t;

// clang-format off

/**
 * @brief Structure describing the set of configurations supported by QNN Graph Transformer.
 *        Objects of this type are to be referenced through QnnGraph_CustomConfig_t.
 *
 *        The struct has two fields - option and a union of corresponding config values.
 *        Based on the option, the corresponding item in the union can be used to specify
 *        the configuration.
 *
 *        Below is the mapping between QnnGraphTransformer_CustomConfigOption_t and config values:
 *
 *        \verbatim embed:rst:leading-asterisk
 *        +----+-------------------------------------------------------------------------+-----------------------------------------------+
 *        | #  | Config Option                                                           | Configuration Value Type                      |
 *        +====+=========================================================================+===============================================+
 *        | 1  | QNN_GRAPH_TRANSFORMER_CUSTOM_CONFIG_OPTION_ENABLE_QNN_GRAPH_TRANSFORMS  | bool enableQnnGraphTransformer                |
 *        +----+-------------------------------------------------------------------------+-----------------------------------------------+
 *        | 2  | QNN_GRAPH_TRANSFORMER_CUSTOM_CONFIG_OPTION_ENABLE_PASSES                | QnnGraphTransformer_PassNames_t enablePasses  |
 *        +----+-------------------------------------------------------------------------+-----------------------------------------------+
 *        | 3  | QNN_GRAPH_TRANSFORMER_CUSTOM_CONFIG_OPTION_DISABLE_PASSES               | QnnGraphTransformer_PassNames_t disablePasses |
 *        +----+-------------------------------------------------------------------------+-----------------------------------------------+
 *        | 4  | QNN_GRAPH_TRANSFORMER_CUSTOM_CONFIG_OPTION_DUMP_DLC                     | bool dumpDLC                                  |
 *        +----+-------------------------------------------------------------------------+-----------------------------------------------+
 *        \endverbatim
 *
 *        Usage Guidelines:
 *        - Option 1: Use boolean value to enable/disable the entire QNN IR optimizer
 *        - Option 2: Use QnnGraphTransformer_PassNames_t to specify passes to enable
 *        - Option 3: Use QnnGraphTransformer_PassNames_t to specify passes to disable
 *        - Options 2 and 3 can be used together for fine-grained pass control
 *        - Option 4: Use boolean value to enable/disable DLC dumping functionality

 */
typedef struct {
  // The configuration option type that determines which union member to use
  QnnGraphTransformer_CustomConfigOption_t option;
  union {
    // Boolean flag to enable/disable the QNN Graph Transformer entirely.
    // Used with QNN_GRAPH_TRANSFORMER_CUSTOM_CONFIG_OPTION_ENABLE_QNN_GRAPH_TRANSFORMS.
    // true = enable optimizer, false = disable optimizer
    bool enableQnnGraphTransformer;
    // List of optimization passes to explicitly enable.
    // Used with QNN_GRAPH_TRANSFORMER_CUSTOM_CONFIG_OPTION_ENABLE_PASSES.
    // Only the specified passes will be enabled during optimization.
    QnnGraphTransformer_PassNames_t enablePasses;
    // List of optimization passes to explicitly disable.
    // Used with QNN_GRAPH_TRANSFORMER_CUSTOM_CONFIG_OPTION_DISABLE_PASSES.
    // The specified passes will be disabled during optimization.
    QnnGraphTransformer_PassNames_t disablePasses;
    // Boolean flag to enable/disable the DLC dumping functionality.
    // Used with QNN_GRAPH_TRANSFORMER_CUSTOM_CONFIG_OPTION_DUMP_DLC.
    bool enableDumpDLC;
  };
} QnnGraphTransformer_CustomConfig_t;

// clang-format on

//=============================================================================
// Initializer Macros
//=============================================================================

// clang-format off

/// QnnGraphTransformer_PassNames_t initializer macro
#define QNN_GRAPH_TRANSFORMER_PASS_NAMES_INIT    \
  {                                              \
    NULL, /*data*/                               \
    0     /*size*/                               \
  }

/// QnnGraphTransformer_CustomConfig_t initializer macro
#define QNN_GRAPH_TRANSFORMER_CUSTOM_CONFIG_INIT                                    \
  {                                                                                 \
    QNN_GRAPH_TRANSFORMER_CUSTOM_CONFIG_OPTION_UNKNOWN, /*option*/                  \
    {                                                                               \
      false /*enableQnnGraphTransformer*/                                          \
    }                                                                               \
  }

/// QnnGraphTransformer_CustomConfig_t initializer macro for DUMP_DLC option
#define QNN_GRAPH_TRANSFORMER_DUMP_DLC_CONFIG_INIT                                  \
  {                                                                                 \
    QNN_GRAPH_TRANSFORMER_CUSTOM_CONFIG_OPTION_DUMP_DLC, /*option*/                 \
    {                                                                               \
      false /*enableDumpDLC*/                                                       \
    }                                                                               \
  }

// clang-format on

#ifdef __cplusplus
}
#endif

// C++ only: runtime options for controlling which optimizer passes are active.
// This struct is used by the adapter layer to pass pass-enable/disable lists
// into QnnGraphTransformer_finalizeGraph.
#ifdef __cplusplus
#include <string>
#include <vector>

namespace qnn {
namespace core {
namespace graphtransformer {

struct QnnGraphTransformerOptions {
  // Passes that the user explicitly wants enabled. All other registered passes
  // will be disabled by default.
  std::vector<std::string> enabledPasses;
  // Passes that the user explicitly wants disabled (in addition to the default
  // all-disabled behaviour). Kept for forward-compatibility.
  std::vector<std::string> disabledPasses;
  // Dump DLC before and after applying optimizations. This is a debug feature.
  bool dumpDLC{false};
};

} // namespace graphtransformer
} // namespace core
} // namespace qnn
#endif // __cplusplus

#endif // QNN_GRAPH_TRANSFORMER_CONFIG_H
