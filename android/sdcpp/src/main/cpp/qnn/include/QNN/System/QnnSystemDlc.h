//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All rights reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

/**
 *  @file
 *  @brief  QNN System Context API.
 *
 *          This is a system API header to provide
 *          Deep Learning Container (DLC) services to users.
 */

#ifndef QNN_SYSTEM_DLC_H
#define QNN_SYSTEM_DLC_H

#include "QnnInterface.h"
#include "QnnTypes.h"
#include "System/QnnSystemCommon.h"
#include "System/QnnSystemContext.h"
#include "System/QnnSystemLog.h"

#ifdef __cplusplus
extern "C" {
#endif

//=============================================================================
// Error Codes
//=============================================================================

/**
 * @brief QNN System Context API result / error codes.
 */
typedef enum {
  QNN_SYSTEM_DLC_MINERROR = QNN_MIN_ERROR_SYSTEM,
  //////////////////////////////////////////

  /// Qnn System Context success
  QNN_SYSTEM_DLC_NO_ERROR = QNN_SYSTEM_COMMON_NO_ERROR,
  /// There is optional API component that is not supported yet.
  QNN_SYSTEM_DLC_ERROR_UNSUPPORTED_FEATURE = QNN_SYSTEM_COMMON_ERROR_UNSUPPORTED_FEATURE,
  /// QNN System DLC invalid handle
  QNN_SYSTEM_DLC_ERROR_INVALID_HANDLE = QNN_SYSTEM_COMMON_ERROR_INVALID_HANDLE,
  /// One or more arguments to a System DLC API is/are NULL/invalid.
  QNN_SYSTEM_DLC_ERROR_INVALID_ARGUMENT = QNN_SYSTEM_COMMON_ERROR_INVALID_ARGUMENT,
  /// Generic Failure in achieving the objective of a System DLC API
  QNN_SYSTEM_DLC_ERROR_OPERATION_FAILED = QNN_SYSTEM_DLC_MINERROR + 2,

  /// Malformed DLC Binary
  QNN_SYSTEM_DLC_ERROR_MALFORMED_BINARY = QNN_SYSTEM_DLC_MINERROR + 10,
  //////////////////////////////////////////
  QNN_SYSTEM_DLC_MAXERROR = QNN_MAX_ERROR_SYSTEM
} QnnSystemDlc_Error_t;

//=============================================================================
// Data Types
//=============================================================================

/// Version of the graph config info
typedef enum {
  QNN_SYSTEM_DLC_GRAPH_CONFIG_INFO_VERSION_1 = 0x01,
  // Unused, present to ensure 32 bits.
  QNN_SYSTEM_DLC_GRAPH_CONFIG_INFO_UNDEFINED = 0x7FFFFFFF
} QnnSystemContext_GraphConfigInfoVersion_t;

typedef struct {
  const char* graphName;
  const QnnGraph_Config_t** graphConfigs;
  uint32_t numConfigs;
} QnnSystemDlc_GraphConfigInfoV1_t;

/// @brief structure to define
typedef struct {
  QnnSystemContext_GraphConfigInfoVersion_t version;
  union UNNAMED {
    QnnSystemDlc_GraphConfigInfoV1_t v1;
  };
} QnnSystemDlc_GraphConfigInfo_t;

//=============================================================================
// Record Types
//=============================================================================
typedef enum {
  // DLC.metadata that stores DLC meta info
  QNN_SYSTEM_DLC_RECORD_TYPE_DLC_METADATA = 0x01,
  // The irGraph topology
  QNN_SYSTEM_DLC_RECORD_TYPE_MODEL_TOPOLOGY = 0x02,
  // The meta info of all the weights of the models
  QNN_SYSTEM_DLC_RECORD_TYPE_MODEL_PARAMS = 0x03,
  // The raw weights of the models
  QNN_SYSTEM_DLC_RECORD_TYPE_MODEL_RAW_WIEGHTS = 0x04,
  // Correctly-spelled alias for QNN_SYSTEM_DLC_RECORD_TYPE_MODEL_RAW_WIEGHTS
  QNN_SYSTEM_DLC_RECORD_TYPE_MODEL_RAW_WEIGHTS = QNN_SYSTEM_DLC_RECORD_TYPE_MODEL_RAW_WIEGHTS,
  // The encodings of all the weights of the models
  QNN_SYSTEM_DLC_RECORD_TYPE_MODEL_ENCODINGS = 0x05,
  // The metatdata of transforms occurring on lora static tensors
  QNN_SYSTEM_DLC_RECORD_TYPE_LORA_CONVERTER_METADATA = 0x06,
  QNN_SYSTEM_DLC_RECORD_PREFIX_HTP_CACHE_RECORD      = 0x07,
  QNN_SYSTEM_DLC_RECORD_PREFIX_AIP_CACHE_RECORD      = 0x08,
  QNN_SYSTEM_DLC_RECORD_PREFIX_HTA_CACHE_RECORD      = 0x09,
  QNN_SYSTEM_DLC_RECORD_PREFIX_AIX_CACHE_RECORD      = 0x0A,
  QNN_SYSTEM_DLC_RECORD_PREFIX_GPU_CACHE_RECORD      = 0x0B,
  QNN_SYSTEM_DLC_RECORD_PREFIX_HEXNNV2_CACHE_RECORD  = 0x0C,
  QNN_SYSTEM_DLC_RECORD_TYPE_SOURCE_MAPPING          = 0x0D,
  QNN_SYSTEM_DLC_RECORD_TYPE_SOURCE_TOPOLOGY         = 0x0E,
  // Record to store the schematic of graph composed on HTP backend
  QNN_SYSTEM_DLC_RECORD_TYPE_HTP_SCHEMATIC_BIN = 0x0F,
  // Record to store the mapping information of ops in irGraph and HTP backend
  QNN_SYSTEM_DLC_RECORD_TYPE_HTP_GRAPH_MAPPING = 0x10,
  QNN_SYSTEM_DLC_RECORD_TYPE_GENAI_METADATA    = 0x11,
  QNN_SYSTEM_DLC_RECORD_TYPE_GENAI_ARTIFACT    = 0x12,
  // Record to store the HTP shared weights
  QNN_SYSTEM_DLC_RECORD_TYPE_HTP_SHARED_WEIGHTS_RECORD = 0x13,
  // Record to store the HTP shared weights metadata
  QNN_SYSTEM_DLC_RECORD_TYPE_HTP_SHARED_WEIGHTS_METADATA_RECORD = 0x14,
  // DLC.metadata.history that stores DLC history info (converter/quantizer/deploy commands)
  QNN_SYSTEM_DLC_RECORD_TYPE_DLC_METADATA_HISTORY = 0x15,
  // Unused, present to ensure 32 bits.
  QNN_SYSTEM_DLC_RECORD_NAME_UNKNOWN = 0x7FFFFFFF
} QnnSystemDlc_RecordType_t;

/**
 * @brief A typedef to indicate a QNN System DLC Record handle
 */
typedef void* QnnSystemDlc_RecordHandle_t;

/**
 * @brief Opaque handle to a composable graph info entry in a DLC.
 *        Lifetime is tied to the owning dlcHandle; freed when dlcHandle is freed.
 */
typedef void* QnnSystemDlc_GraphInfoHandle_t;

/**
 * @brief Opaque handle to a prepared context-binary entry in a DLC.
 *        Lifetime is tied to the owning dlcHandle; freed when dlcHandle is freed.
 */
typedef void* QnnSystemDlc_ContextBinaryInfoHandle_t;

/**
 * @brief Opaque handle to a single graph entry within a context binary.
 *        Provides read-only access to graph metadata (name, tensor counts, tensor info).
 *        Lifetime is tied to the owning contextBinaryInfoHandle (and transitively to dlcHandle).
 *        Do not free this handle directly.
 */
typedef void* QnnSystemDlc_ContextGraphInfoHandle_t;

/**
 * @brief Opaque handle to tensor info (name, type, shape, quant params) for a single tensor
 *        within a graph.  Provides read-only access to tensor metadata; holds no tensor data.
 *        Lifetime is tied to the owning graphInfoHandle (and transitively to dlcHandle).
 *        Do not free this handle directly; it is invalidated when the parent dlcHandle is freed.
 */
typedef void* QnnSystemDlc_TensorInfoHandle_t;

/**
 * @brief Opaque handle to the quantization parameters of a tensor.
 *        Lifetime is tied to the owning tensorInfoHandle.
 *        Do not free this handle directly.
 */
typedef void* QnnSystemDlc_QuantParamsHandle_t;

/// Version of the composable graph info structure
typedef enum {
  QNN_SYSTEM_DLC_COMPOSABLE_GRAPH_INFO_VERSION_1 = 0x01,
  // Unused, present to ensure 32 bits.
  QNN_SYSTEM_DLC_COMPOSABLE_GRAPH_INFO_UNDEFINED = 0x7FFFFFFF
} QnnSystemDlc_ComposableGraphInfoVersion_t;

/**
 * @brief Struct that provides input/output tensor metadata for a composable (IR) graph in a DLC.
 *        This is version V1 of the structure.
 */
typedef struct {
  /// Name of graph
  const char* graphName;
  /// Number of input tensors to graph
  uint32_t numGraphInputs;
  /// List of input tensors to graph
  Qnn_Tensor_t* graphInputs;
  /// Number of output tensors from graph
  uint32_t numGraphOutputs;
  /// List of output tensors from graph
  Qnn_Tensor_t* graphOutputs;
} QnnSystemDlc_ComposableGraphInfoV1_t;

// clang-format off
/// QnnSystemDlc_ComposableGraphInfoV1_t initializer macro
#define QNN_SYSTEM_DLC_COMPOSABLE_GRAPH_INFO_V1_INIT \
  {                                                   \
    NULL, /* graphName */                             \
    0,    /* numGraphInputs */                        \
    NULL, /* graphInputs */                           \
    0,    /* numGraphOutputs */                       \
    NULL, /* graphOutputs */                          \
  }
// clang-format on

typedef struct {
  QnnSystemDlc_ComposableGraphInfoVersion_t version;
  union UNNAMED {
    QnnSystemDlc_ComposableGraphInfoV1_t graphInfoV1;
  };
} QnnSystemDlc_ComposableGraphInfo_t;

// clang-format off
/// QnnSystemDlc_ComposableGraphInfo_t initializer macro
#define QNN_SYSTEM_DLC_COMPOSABLE_GRAPH_INFO_INIT          \
  {                                                         \
    QNN_SYSTEM_DLC_COMPOSABLE_GRAPH_INFO_UNDEFINED,        \
    {                                                       \
      QNN_SYSTEM_DLC_COMPOSABLE_GRAPH_INFO_V1_INIT         \
    }                                                       \
  }
// clang-format on

//=============================================================================
// Public Functions
//=============================================================================

/**
 * @brief A function to create an instance of the DLC from a file
 *
 * @param[in] dlcPath path the DLC
 * @param[in] logger a log handle produced from QnnSystemLog_create(). Can be NULL
 * @param[out] dlcHandle A handle to the created instance of a systemDlc entity
 *
 * @return Error code
 *         - QNN_SUCCESS: Successfully created a systemDlc entity
 *         - QNN_SYSTEM_DLC_ERROR_INVALID_ARGUMENT: sysCtxHandle is NULL
 *         - QNN_COMMON_ERROR_MEM_ALLOC: Error encountered in allocating memory for
 *           systemDlc instance
 *         - QNN_SYSTEM_DLC_ERROR_UNSUPPORTED_FEATURE: system context features not supported
 */
QNN_SYSTEM_API
Qnn_ErrorHandle_t QnnSystemDlc_createFromFile(Qnn_LogHandle_t logger,
                                              const char* dlcPath,
                                              QnnSystemDlc_Handle_t* dlcHandle);

/**
 * @brief A function to create an instance of the DLC from a file with a destination directory hint
 *
 * @param[in] logger a log handle produced from QnnSystemLog_create(). Can be NULL
 * @param[in] dlcPath path to the DLC
 * @param[in] destinationDirectoryHint path to a directory where the DLC will be saved after
 *            changes. Must not be NULL.
 * @param[out] dlcHandle A handle to the created instance of a systemDlc entity
 *
 * @return Error code
 *         - QNN_SUCCESS: Successfully created a systemDlc entity
 *         - QNN_SYSTEM_DLC_ERROR_INVALID_ARGUMENT: dlcPath, destinationDirectoryHint, or
 *           dlcHandle is NULL
 *         - QNN_COMMON_ERROR_MEM_ALLOC: Error encountered in allocating memory for
 *           systemDlc instance
 *         - QNN_SYSTEM_DLC_ERROR_UNSUPPORTED_FEATURE: system context features not supported
 */
QNN_SYSTEM_API
Qnn_ErrorHandle_t QnnSystemDlc_createFromFileWithDestinationDir(
    Qnn_LogHandle_t logger,
    const char* dlcPath,
    const char* destinationDirectoryHint,
    QnnSystemDlc_Handle_t* dlcHandle);

/**
 * @brief A function to create an instance of the DLC from a binary buffer
 *
 * @param[in]  buffer pointer to buffer representing the DLC
 * @param[in]  logger a log handle produced from QnnSystemLog_create(). Can be NULL
 * @param[in]  bufferSize size of the binary buffer
 * @param[out] dlcHandle A handle to the created instance of a systemDlc entity
 *
 * @return Error code
 *         - QNN_SUCCESS: Successfully created a systemDlc entity
 *         - QNN_SYSTEM_DLC_ERROR_INVALID_ARGUMENT: sysCtxHandle is NULL
 *         - QNN_COMMON_ERROR_MEM_ALLOC: Error encountered in allocating memory for
 *           systemDlc instance
 *         - QNN_SYSTEM_DLC_ERROR_UNSUPPORTED_FEATURE: system context features not supported
 */
QNN_SYSTEM_API
Qnn_ErrorHandle_t QnnSystemDlc_createFromBinary(Qnn_LogHandle_t logger,
                                                const uint8_t* buffer,
                                                const Qnn_ContextBinarySize_t bufferSize,
                                                QnnSystemDlc_Handle_t* dlcHandle);

/**
 * @brief A function to create an instance of the DLC from a binary buffer with a destination
 *        directory hint
 *
 * @param[in]  logger a log handle produced from QnnSystemLog_create(). Can be NULL
 * @param[in]  buffer pointer to buffer representing the DLC
 * @param[in]  bufferSize size of the binary buffer
 * @param[in]  destinationDirectoryHint path to a directory where the DLC will be saved after
 *             changes. Must not be NULL.
 * @param[out] dlcHandle A handle to the created instance of a systemDlc entity
 *
 * @return Error code
 *         - QNN_SUCCESS: Successfully created a systemDlc entity
 *         - QNN_SYSTEM_DLC_ERROR_INVALID_ARGUMENT: buffer, destinationDirectoryHint, or
 *           dlcHandle is NULL
 *         - QNN_COMMON_ERROR_MEM_ALLOC: Error encountered in allocating memory for
 *           systemDlc instance
 *         - QNN_SYSTEM_DLC_ERROR_UNSUPPORTED_FEATURE: system context features not supported
 */
QNN_SYSTEM_API
Qnn_ErrorHandle_t QnnSystemDlc_createFromBinaryWithDestinationDir(
    Qnn_LogHandle_t logger,
    const uint8_t* buffer,
    const Qnn_ContextBinarySize_t bufferSize,
    const char* destinationDirectoryHint,
    QnnSystemDlc_Handle_t* dlcHandle);

/**
 * @brief A function to compose graphs from a DLC on a particular backend, __backend__, through
 *        an backendInterface __backendInterface__. Memory allocated in __graphs__ is owned by clients and may
 *        be released with calls to free().
 *
 * @param[in]  dlcHandle the DLC to retrieve graphs from
 * @param[in]  graphConfigs the graph configuration information for a particular graph
 * @param[in]  numGraphConfigs number of graph configurations
 * @param[in] backend the backend on which to compose the graphs
 * @param[in]  context the context on which to compose the graphs
 * @param[in]  backendInterface the interface used to compose the graph.
 * @param[in] graphVersion version of the graph info structure to be returned
 * @param[out] graphs An array of graph information representing what was created with the backend
 * @param[out] numGraphs the number of created graphs
 *
 * @return Error code
 *         - QNN_SUCCESS: Successfully composed graphs
 *         - QNN_SYSTEM_DLC_ERROR_INVALID_ARGUMENT: Argument is NULL
 *         - QNN_COMMON_ERROR_MEM_ALLOC: Error encountered in allocating memory for
 *         - QNN_SYSTEM_DLC_ERROR_INVALID_HANDLE: Invalid Dlc handle to free
 *         - QNN_SYSTEM_DLC_ERROR_UNSUPPORTED_FEATURE: DLC features not supported
 *
 */
QNN_SYSTEM_API
Qnn_ErrorHandle_t QnnSystemDlc_composeGraphs(QnnSystemDlc_Handle_t dlcHandle,
                                             const QnnSystemDlc_GraphConfigInfo_t** graphConfigs,
                                             const uint32_t numGraphConfigs,
                                             Qnn_BackendHandle_t backend,
                                             Qnn_ContextHandle_t context,
                                             QnnInterface_t backendInterface,
                                             QnnSystemContext_GraphInfoVersion_t graphVersion,
                                             QnnSystemContext_GraphInfo_t** graphs,
                                             uint32_t* numGraphs);
/**
 * @brief A function to retrieve Op Mapping information from a DLC
 *
 * @param[in]  dlcHandle Handle to the DLC
 * @param[out] opMappings a list of op mappings. The memory allocated here is owned by the System
 *             library and is released when the corresponding DLC Handle is freed
 * @param[out] numOpMappings the number of opMappings
 *
 * @return Error code
 *         - QNN_SUCCESS: Successfully freed instance of System Context
 *         - QNN_SYSTEM_DLC_ERROR_INVALID_HANDLE: Invalid Dlc handle to free
 *         - QNN_SYSTEM_DLC_ERROR_UNSUPPORTED_FEATURE: not supported
 */
QNN_SYSTEM_API
Qnn_ErrorHandle_t QnnSystemDlc_getOpMappings(QnnSystemDlc_Handle_t dlcHandle,
                                             const Qnn_OpMapping_t** opMappings,
                                             uint32_t* numOpMappings);

/**
 * @brief A function to retrieve a record associated with handle __dlcHandle__
 *        of name __recordName__
 *
 * @param[in] dlcHandle handle to the DLC to which this record is associated
 * @param[in] recordName the name of the record to retrieve
 * @param[out] recordHandle record handle matching __recordName__
 *
 * @note If there are no records that match __recordName__, then recordHandle will
 *       be set to nullptr. Record names are unique within a DLC so this API will only
 *       return at most one record handle
 *
 * @return Error code
 *         - QNN_SUCCESS: Successfully retrieved record of __recordName__
 *         - QNN_SYSTEM_DLC_ERROR_INVALID_HANDLE: Invalid dlcHandle passed
 *         - QNN_SYSTEM_DLC_ERROR_INVALID_ARGUMENT: Invalid recordName or recordHandle passed
 */
QNN_SYSTEM_API
Qnn_ErrorHandle_t QnnSystemDlc_getRecordByName(QnnSystemDlc_Handle_t dlcHandle,
                                               const char* recordName,
                                               QnnSystemDlc_RecordHandle_t* recordHandle);

/**
 * @brief A function to retrieve records associated with handle __dlcHandle__ of type
 *        __recordType__
 *
 * @param[in] dlcHandle handle to the DLC to which the records are associated
 * @param[in] recordType the type of the records to retrieve
 * @param[in] getMostOptimalContextBinary option to retrieve the most-compatible context binary on
 *                                        current SoC. This option is only useful when retrieving
 *                                        context binaries. If set to 1, no more than one context
 *                                        binary will be returned even if the DLC has multiple
 *                                        context binaries that match the provided __recordType__.
 *                                        If set to 0, all context binaries that match __recordType__
 *                                        will be returned
 * @param[out] recordHandles array of record handles matching __recordType__
 * @param[out] numRecordHandles number of record handles in recordHandles array
 *
 * @note If there are no records of __recordType__, then recordHandles will be set
 *       to nullptr and numRecordHandles will be set to 0
 *
 * @return Error code
 *         - QNN_SUCCESS: Successfully retrieving records of type __recordType__
 *         - QNN_SYSTEM_DLC_ERROR_INVALID_HANDLE: Invalid dlcHandle passed
 *         - QNN_SYSTEM_DLC_ERROR_INVALID_ARGUMENT: invalid recordHandles or numRecordHandles passed
 */
QNN_SYSTEM_API
Qnn_ErrorHandle_t QnnSystemDlc_getRecordsByType(QnnSystemDlc_Handle_t dlcHandle,
                                                QnnSystemDlc_RecordType_t recordType,
                                                uint8_t getMostOptimalContextBinary,
                                                QnnSystemDlc_RecordHandle_t** recordHandles,
                                                uint32_t* numRecordHandles);

/**
 * @brief A function to retrieve the size of the data in __recordHandle__
 *
 * @param[in] recordHandle record handle to retrieve the size of the data from
 * @param[out] dataSize size of the data in the record
 *
 * @return Error code
 *         - QNN_SUCCESS: Successfully retrieved the size of the record
 *         - QNN_SYSTEM_DLC_ERROR_INVALID_HANDLE: Invalid record handle
 *         - QNN_SYSTEM_DLC_ERROR_INVALID_ARGUMENT: invalid dataSize pointer passed
 */
QNN_SYSTEM_API
Qnn_ErrorHandle_t QnnSystemDlc_getRecordDataSize(QnnSystemDlc_RecordHandle_t recordHandle,
                                                 uint64_t* dataSize);

/**
 * @brief A function to retrieve the content of __recordHandle__ as a memory mapped buffer __data__
 *
 * @param[in] recordHandle record handle to read from
 * @param[out] data data read from the record
 * @param[out] dataSize size of the data read
 *
 * @return Error code
 *         - QNN_SUCCESS: Successfully read the data of the record to user buffer
 *         - QNN_SYSTEM_DLC_ERROR_INVALID_HANDLE: Invalid record handle
 *         - QNN_SYSTEM_DLC_ERROR_INVALID_ARGUMENT: invalid user data pointer or dataSize
 *                                                  pointer passed
 */
QNN_SYSTEM_API
Qnn_ErrorHandle_t QnnSystemDlc_readRecordDataMemoryMapped(QnnSystemDlc_RecordHandle_t recordHandle,
                                                          const uint8_t** data,
                                                          uint64_t* dataSize);

/**
 * @brief A function to free a record. Record will also be freed if the associated
 *        DLC handle is freed
 *
 * @param[in] recordHandle handle to the record to be freed
 *
 * @note If the record is associated with a DLC, this API does not remove the record
 *       from the DLC. It will only free the record handle. To remove the record from
 *       the DLC, use the QnnSystemDlc_removeRecordByType() API
 *
 * @note This API will fail if the record is associated with a DLC and its handle has
 *       already been freed. Since freeing the DLC handle also frees all handles of
 *       the records associated with it, this operation will result in failure
 *
 * @return Error code
 *         - QNN_SUCCESS: Successfully freed instance of record handle
 *         - QNN_SYSTEM_DLC_ERROR_INVALID_HANDLE: Invalid record handle to free
 */
QNN_SYSTEM_API
Qnn_ErrorHandle_t QnnSystemDlc_freeRecord(QnnSystemDlc_RecordHandle_t recordHandle);

/**
 *  @brief Free a contiguous memory range within the record's memory allocation.
 *
 * @param[in] recordHandle record
 * @param[in] offset Byte offset from the start of the record's memory allocation
 * @param[in] size Size in bytes of the memory range to free
 *
 * @return Error code
 *         - QNN_SUCCESS: Successfully freed the memory range
 *         - QNN_SYSTEM_DLC_ERROR_INVALID_HANDLE: Invalid record handle
 *         - QNN_SYSTEM_DLC_ERROR_INVALID_ARGUMENT: size is 0
 */
QNN_SYSTEM_API
Qnn_ErrorHandle_t QnnSystemDlc_freeRecordRange(QnnSystemDlc_RecordHandle_t recordHandle,
                                               uint64_t offset,
                                               uint64_t size);

/**
 * @brief Returns handles for all composable graphs present in the DLC.
 *
 * Each returned handle can be passed to QnnSystemDlc_getComposableGraphInfo to inspect
 * the graph's tensor metadata without composing it on a backend.
 *
 * @param[in]  dlcHandle            Handle to the DLC object.
 * @param[out] graphInfoHandles     Array of composable graph info handles.
 *                                  Memory owned by the DLC system library; valid until
 *                                  dlcHandle is freed.
 * @param[out] numGraphInfoHandles  Number of handles returned.
 *
 * @return Error code
 *         - QNN_SUCCESS: Successfully retrieved handles (numGraphInfoHandles may be 0 if the
 *           DLC contains no composable graphs, in which case graphInfoHandles is set to NULL).
 *         - QNN_SYSTEM_DLC_ERROR_INVALID_HANDLE: Invalid dlcHandle.
 *         - QNN_SYSTEM_DLC_ERROR_INVALID_ARGUMENT: graphInfoHandles or numGraphInfoHandles is NULL.
 *         - QNN_SYSTEM_DLC_ERROR_OPERATION_FAILED: Failed to load graph metadata.
 */
QNN_SYSTEM_API
Qnn_ErrorHandle_t QnnSystemDlc_getComposableGraphInfoHandles(
    QnnSystemDlc_Handle_t dlcHandle,
    QnnSystemDlc_GraphInfoHandle_t** graphInfoHandles,
    uint32_t* numGraphInfoHandles);

/**
 * @brief Returns handles for all context binaries present in the DLC.
 *
 * Each returned handle can be passed to QnnSystemDlc_getContextBinaryInfo to inspect
 * the metadata of the corresponding backend-specific context binary (e.g. HTP/AIP cache records).
 *
 * @param[in]  dlcHandle                    Handle to the DLC object.
 * @param[out] contextBinaryInfoHandles     Array of context info handles.
 *                                          Memory owned by the DLC system library; valid until
 *                                          dlcHandle is freed.
 * @param[out] numContextBinaryInfoHandles  Number of handles returned.
 *
 * @return Error code
 *         - QNN_SUCCESS: Successfully retrieved handles (numContextBinaryInfoHandles may be 0 if
 *           the DLC contains no context binaries, in which case contextBinaryInfoHandles is set to
 *           NULL).
 *         - QNN_SYSTEM_DLC_ERROR_INVALID_HANDLE: Invalid dlcHandle.
 *         - QNN_SYSTEM_DLC_ERROR_INVALID_ARGUMENT: contextBinaryInfoHandles or numContextBinaryInfoHandles is
 *           NULL.
 *         - QNN_SYSTEM_DLC_ERROR_OPERATION_FAILED: Failed to retrieve context binary info.
 */
QNN_SYSTEM_API
Qnn_ErrorHandle_t QnnSystemDlc_getContextBinaryInfoHandles(
    QnnSystemDlc_Handle_t dlcHandle,
    QnnSystemDlc_ContextBinaryInfoHandle_t** contextBinaryInfoHandles,
    uint32_t* numContextBinaryInfoHandles);

/**
 * @brief Returns graph tensor metadata for a composable graph handle.
 *
 * Exposes the graph's input/output tensor signature without requiring a backend compose step.
 *
 * @param[in]  graphInfoHandle  Handle obtained from QnnSystemDlc_getComposableGraphInfoHandles.
 * @param[out] graphInfo        Populated graph info struct.
 *                              Memory owned by the DLC system library; valid until dlcHandle freed.
 *
 * @return Error code
 *         - QNN_SUCCESS: Successfully populated graphInfo.
 *         - QNN_SYSTEM_DLC_ERROR_INVALID_HANDLE: Invalid graphInfoHandle.
 *         - QNN_SYSTEM_DLC_ERROR_INVALID_ARGUMENT: graphInfo is NULL.
 *         - QNN_SYSTEM_DLC_ERROR_OPERATION_FAILED: Failed to parse graph metadata.
 */
QNN_SYSTEM_API
Qnn_ErrorHandle_t QnnSystemDlc_getComposableGraphInfo(
    QnnSystemDlc_GraphInfoHandle_t graphInfoHandle,
    const QnnSystemDlc_ComposableGraphInfo_t** graphInfo);

/**
 * @brief Returns the graph name for a composable graph info handle.
 *
 * @param[in]  graphInfoHandle  Handle obtained from QnnSystemDlc_getComposableGraphInfoHandles.
 * @param[out] graphName        Null-terminated graph name string. Memory owned by the library;
 *                              valid until dlcHandle is freed.
 *
 * @return Error code
 *         - QNN_SUCCESS: Successfully retrieved graph name.
 *         - QNN_SYSTEM_DLC_ERROR_INVALID_HANDLE: Invalid graphInfoHandle.
 *         - QNN_SYSTEM_DLC_ERROR_INVALID_ARGUMENT: graphName is NULL.
 */
QNN_SYSTEM_API
Qnn_ErrorHandle_t QnnSystemDlc_getComposableGraphInfoGraphName(
    QnnSystemDlc_GraphInfoHandle_t graphInfoHandle, const char** graphName);

/**
 * @brief Returns the number of input tensors for a composable graph info handle.
 *
 * @param[in]  graphInfoHandle  Handle obtained from QnnSystemDlc_getComposableGraphInfoHandles.
 * @param[out] numInputTensors  Number of input tensors.
 *
 * @return Error code
 *         - QNN_SUCCESS: Successfully retrieved count.
 *         - QNN_SYSTEM_DLC_ERROR_INVALID_HANDLE: Invalid graphInfoHandle.
 *         - QNN_SYSTEM_DLC_ERROR_INVALID_ARGUMENT: numInputTensors is NULL.
 */
QNN_SYSTEM_API
Qnn_ErrorHandle_t QnnSystemDlc_getComposableGraphInfoNumInputTensors(
    QnnSystemDlc_GraphInfoHandle_t graphInfoHandle, uint32_t* numInputTensors);

/**
 * @brief Returns the number of output tensors for a composable graph info handle.
 *
 * @param[in]  graphInfoHandle   Handle obtained from QnnSystemDlc_getComposableGraphInfoHandles.
 * @param[out] numOutputTensors  Number of output tensors.
 *
 * @return Error code
 *         - QNN_SUCCESS: Successfully retrieved count.
 *         - QNN_SYSTEM_DLC_ERROR_INVALID_HANDLE: Invalid graphInfoHandle.
 *         - QNN_SYSTEM_DLC_ERROR_INVALID_ARGUMENT: numOutputTensors is NULL.
 */
QNN_SYSTEM_API
Qnn_ErrorHandle_t QnnSystemDlc_getComposableGraphInfoNumOutputTensors(
    QnnSystemDlc_GraphInfoHandle_t graphInfoHandle, uint32_t* numOutputTensors);

/**
 * @brief Returns a handle to the input tensor at the given zero-based index.
 *
 * Call QnnSystemDlc_getComposableGraphInfoNumInputTensors first to determine valid index range.
 * The returned handle is valid until dlcHandle is freed.
 *
 * @param[in]  graphInfoHandle  Handle obtained from QnnSystemDlc_getComposableGraphInfoHandles.
 * @param[in]  index            Zero-based index of the input tensor.
 * @param[out] tensorHandle     Handle to the requested input tensor.
 *
 * @return Error code
 *         - QNN_SUCCESS: Successfully retrieved tensor handle.
 *         - QNN_SYSTEM_DLC_ERROR_INVALID_HANDLE: Invalid graphInfoHandle.
 *         - QNN_SYSTEM_DLC_ERROR_INVALID_ARGUMENT: tensorHandle is NULL or index is out of range.
 */
QNN_SYSTEM_API
Qnn_ErrorHandle_t QnnSystemDlc_getGraphInfoInputTensorInfo(
    QnnSystemDlc_GraphInfoHandle_t graphInfoHandle,
    uint32_t index,
    QnnSystemDlc_TensorInfoHandle_t* tensorHandle);

/**
 * @brief Returns a handle to the output tensor at the given zero-based index.
 *
 * Call QnnSystemDlc_getComposableGraphInfoNumOutputTensors first to determine valid index range.
 * The returned handle is valid until dlcHandle is freed.
 *
 * @param[in]  graphInfoHandle  Handle obtained from QnnSystemDlc_getComposableGraphInfoHandles.
 * @param[in]  index            Zero-based index of the output tensor.
 * @param[out] tensorHandle     Handle to the requested output tensor.
 *
 * @return Error code
 *         - QNN_SUCCESS: Successfully retrieved tensor handle.
 *         - QNN_SYSTEM_DLC_ERROR_INVALID_HANDLE: Invalid graphInfoHandle.
 *         - QNN_SYSTEM_DLC_ERROR_INVALID_ARGUMENT: tensorHandle is NULL or index is out of range.
 */
QNN_SYSTEM_API
Qnn_ErrorHandle_t QnnSystemDlc_getGraphInfoOutputTensorInfo(
    QnnSystemDlc_GraphInfoHandle_t graphInfoHandle,
    uint32_t index,
    QnnSystemDlc_TensorInfoHandle_t* tensorHandle);

/**
 * @brief Returns the name of a tensor.
 *
 * @param[in]  tensorHandle  Handle obtained from QnnSystemDlc_getGraphInfoInputTensorInfo or
 *                           QnnSystemDlc_getGraphInfoOutputTensorInfo.
 * @param[out] name          Null-terminated tensor name. Memory owned by the library.
 *
 * @return Error code
 *         - QNN_SUCCESS: Successfully retrieved tensor name.
 *         - QNN_SYSTEM_DLC_ERROR_INVALID_HANDLE: Invalid tensorHandle.
 *         - QNN_SYSTEM_DLC_ERROR_INVALID_ARGUMENT: name is NULL.
 */
QNN_SYSTEM_API
Qnn_ErrorHandle_t QnnSystemDlc_getTensorInfoName(QnnSystemDlc_TensorInfoHandle_t tensorHandle,
                                             const char** name);

/**
 * @brief Returns the integer ID of a tensor.
 *
 * @param[in]  tensorHandle  Tensor handle.
 * @param[out] id            Tensor ID.
 *
 * @return Error code
 *         - QNN_SUCCESS: Successfully retrieved tensor ID.
 *         - QNN_SYSTEM_DLC_ERROR_INVALID_HANDLE: Invalid tensorHandle.
 *         - QNN_SYSTEM_DLC_ERROR_INVALID_ARGUMENT: id is NULL.
 */
QNN_SYSTEM_API
Qnn_ErrorHandle_t QnnSystemDlc_getTensorInfoId(QnnSystemDlc_TensorInfoHandle_t tensorHandle,
                                           uint32_t* id);

/**
 * @brief Returns the type of a tensor.
 *
 * @param[in]  tensorHandle  Tensor handle.
 * @param[out] type          Tensor type (see Qnn_TensorType_t).
 *
 * @return Error code
 *         - QNN_SUCCESS: Successfully retrieved tensor type.
 *         - QNN_SYSTEM_DLC_ERROR_INVALID_HANDLE: Invalid tensorHandle.
 *         - QNN_SYSTEM_DLC_ERROR_INVALID_ARGUMENT: type is NULL.
 */
QNN_SYSTEM_API
Qnn_ErrorHandle_t QnnSystemDlc_getTensorInfoType(QnnSystemDlc_TensorInfoHandle_t tensorHandle,
                                             Qnn_TensorType_t* type);

/**
 * @brief Returns the data format of a tensor.
 *
 * @param[in]  tensorHandle  Tensor handle.
 * @param[out] dataFormat    Tensor data format (see Qnn_TensorDataFormat_t).
 *
 * @return Error code
 *         - QNN_SUCCESS: Successfully retrieved tensor data format.
 *         - QNN_SYSTEM_DLC_ERROR_INVALID_HANDLE: Invalid tensorHandle.
 *         - QNN_SYSTEM_DLC_ERROR_INVALID_ARGUMENT: dataFormat is NULL.
 */
QNN_SYSTEM_API
Qnn_ErrorHandle_t QnnSystemDlc_getTensorInfoDataFormat(QnnSystemDlc_TensorInfoHandle_t tensorHandle,
                                                   Qnn_TensorDataFormat_t* dataFormat);

/**
 * @brief Returns the data type of a tensor.
 *
 * @param[in]  tensorHandle  Tensor handle.
 * @param[out] dataType      Tensor data type (see Qnn_DataType_t).
 *
 * @return Error code
 *         - QNN_SUCCESS: Successfully retrieved tensor data type.
 *         - QNN_SYSTEM_DLC_ERROR_INVALID_HANDLE: Invalid tensorHandle.
 *         - QNN_SYSTEM_DLC_ERROR_INVALID_ARGUMENT: dataType is NULL.
 */
QNN_SYSTEM_API
Qnn_ErrorHandle_t QnnSystemDlc_getTensorInfoDataType(QnnSystemDlc_TensorInfoHandle_t tensorHandle,
                                                 Qnn_DataType_t* dataType);

/**
 * @brief Returns the rank (number of dimensions) of a tensor.
 *
 * @param[in]  tensorHandle  Tensor handle.
 * @param[out] rank          Tensor rank.
 *
 * @return Error code
 *         - QNN_SUCCESS: Successfully retrieved tensor rank.
 *         - QNN_SYSTEM_DLC_ERROR_INVALID_HANDLE: Invalid tensorHandle.
 *         - QNN_SYSTEM_DLC_ERROR_INVALID_ARGUMENT: rank is NULL.
 */
QNN_SYSTEM_API
Qnn_ErrorHandle_t QnnSystemDlc_getTensorInfoRank(QnnSystemDlc_TensorInfoHandle_t tensorHandle,
                                             uint32_t* rank);

/**
 * @brief Fills a caller-allocated array with the dimensions of a tensor.
 *
 * The caller must first call QnnSystemDlc_getTensorInfoRank to obtain the rank, then allocate a
 * uint32_t array of that length before calling this function.
 *
 * @param[in]  tensorHandle  Tensor handle.
 * @param[out] dimensions    Caller-allocated array of length rank to receive dimension values.
 *
 * @return Error code
 *         - QNN_SUCCESS: Successfully filled dimensions array.
 *         - QNN_SYSTEM_DLC_ERROR_INVALID_HANDLE: Invalid tensorHandle.
 *         - QNN_SYSTEM_DLC_ERROR_INVALID_ARGUMENT: dimensions is NULL.
 */
QNN_SYSTEM_API
Qnn_ErrorHandle_t QnnSystemDlc_getTensorInfoDimensions(QnnSystemDlc_TensorInfoHandle_t tensorHandle,
                                                   uint32_t* dimensions);

/**
 * @brief Returns a handle to the quantization parameters of a tensor.
 *
 * The returned handle is valid until dlcHandle is freed. Do not free this handle directly.
 *
 * @param[in]  tensorHandle      Tensor handle.
 * @param[out] quantParamsHandle Handle to the tensor's quantization parameters.
 *
 * @return Error code
 *         - QNN_SUCCESS: Successfully retrieved quantization params handle.
 *         - QNN_SYSTEM_DLC_ERROR_INVALID_HANDLE: Invalid tensorHandle.
 *         - QNN_SYSTEM_DLC_ERROR_INVALID_ARGUMENT: quantParamsHandle is NULL.
 */
QNN_SYSTEM_API
Qnn_ErrorHandle_t QnnSystemDlc_getTensorInfoQuantParams(
    QnnSystemDlc_TensorInfoHandle_t tensorHandle,
    QnnSystemDlc_QuantParamsHandle_t* quantParamsHandle);

/**
 * @brief Returns the encoding definition (whether quantization is defined) for a
 *        quantization params handle.
 *
 * @param[in]  quantParamsHandle  Handle obtained from QnnSystemDlc_getTensorInfoQuantParams.
 * @param[out] encodingDefinition Encoding definition (see Qnn_Definition_t).
 *
 * @return Error code
 *         - QNN_SUCCESS: Successfully retrieved encoding definition.
 *         - QNN_SYSTEM_DLC_ERROR_INVALID_HANDLE: Invalid quantParamsHandle.
 *         - QNN_SYSTEM_DLC_ERROR_INVALID_ARGUMENT: encodingDefinition is NULL.
 */
QNN_SYSTEM_API
Qnn_ErrorHandle_t QnnSystemDlc_getQuantParamsEncodingDefinition(
    QnnSystemDlc_QuantParamsHandle_t quantParamsHandle, Qnn_Definition_t* encodingDefinition);

/**
 * @brief Returns the active quantization encoding type for a quantization params handle.
 *
 * @param[in]  quantParamsHandle  Handle obtained from QnnSystemDlc_getTensorInfoQuantParams.
 * @param[out] encoding           Active encoding type (see Qnn_QuantizationEncoding_t).
 *
 * @return Error code
 *         - QNN_SUCCESS: Successfully retrieved encoding type.
 *         - QNN_SYSTEM_DLC_ERROR_INVALID_HANDLE: Invalid quantParamsHandle.
 *         - QNN_SYSTEM_DLC_ERROR_INVALID_ARGUMENT: encoding is NULL.
 */
QNN_SYSTEM_API
Qnn_ErrorHandle_t QnnSystemDlc_getQuantParamsEncoding(
    QnnSystemDlc_QuantParamsHandle_t quantParamsHandle, Qnn_QuantizationEncoding_t* encoding);

/**
 * @brief Returns the scale for QNN_QUANTIZATION_ENCODING_SCALE_OFFSET encoding.
 *
 * @param[in]  quantParamsHandle  Handle obtained from QnnSystemDlc_getTensorInfoQuantParams.
 * @param[out] scale              Scale value.
 *
 * @return Error code
 *         - QNN_SUCCESS: Successfully retrieved scale.
 *         - QNN_SYSTEM_DLC_ERROR_INVALID_HANDLE: Invalid quantParamsHandle.
 *         - QNN_SYSTEM_DLC_ERROR_INVALID_ARGUMENT: scale is NULL, or active encoding is not
 *           QNN_QUANTIZATION_ENCODING_SCALE_OFFSET.
 */
QNN_SYSTEM_API
Qnn_ErrorHandle_t QnnSystemDlc_getQuantParamsScaleOffsetScale(
    QnnSystemDlc_QuantParamsHandle_t quantParamsHandle, float* scale);

/**
 * @brief Returns the offset for QNN_QUANTIZATION_ENCODING_SCALE_OFFSET encoding.
 *
 * @param[in]  quantParamsHandle  Handle obtained from QnnSystemDlc_getTensorInfoQuantParams.
 * @param[out] offset             Offset value.
 *
 * @return Error code
 *         - QNN_SUCCESS: Successfully retrieved offset.
 *         - QNN_SYSTEM_DLC_ERROR_INVALID_HANDLE: Invalid quantParamsHandle.
 *         - QNN_SYSTEM_DLC_ERROR_INVALID_ARGUMENT: offset is NULL, or active encoding is not
 *           QNN_QUANTIZATION_ENCODING_SCALE_OFFSET.
 */
QNN_SYSTEM_API
Qnn_ErrorHandle_t QnnSystemDlc_getQuantParamsScaleOffsetOffset(
    QnnSystemDlc_QuantParamsHandle_t quantParamsHandle, int32_t* offset);

/**
 * @brief Returns the bitwidth for QNN_QUANTIZATION_ENCODING_BW_SCALE_OFFSET encoding.
 *
 * @param[in]  quantParamsHandle  Handle obtained from QnnSystemDlc_getTensorInfoQuantParams.
 * @param[out] bitwidth           Bitwidth value.
 *
 * @return Error code
 *         - QNN_SUCCESS: Successfully retrieved bitwidth.
 *         - QNN_SYSTEM_DLC_ERROR_INVALID_HANDLE: Invalid quantParamsHandle.
 *         - QNN_SYSTEM_DLC_ERROR_INVALID_ARGUMENT: bitwidth is NULL, or active encoding is not
 *           QNN_QUANTIZATION_ENCODING_BW_SCALE_OFFSET.
 */
QNN_SYSTEM_API
Qnn_ErrorHandle_t QnnSystemDlc_getQuantParamsBwScaleOffsetBitwidth(
    QnnSystemDlc_QuantParamsHandle_t quantParamsHandle, uint32_t* bitwidth);

/**
 * @brief Returns the scale for QNN_QUANTIZATION_ENCODING_BW_SCALE_OFFSET encoding.
 *
 * @param[in]  quantParamsHandle  Handle obtained from QnnSystemDlc_getTensorInfoQuantParams.
 * @param[out] scale              Scale value.
 *
 * @return Error code
 *         - QNN_SUCCESS: Successfully retrieved scale.
 *         - QNN_SYSTEM_DLC_ERROR_INVALID_HANDLE: Invalid quantParamsHandle.
 *         - QNN_SYSTEM_DLC_ERROR_INVALID_ARGUMENT: scale is NULL, or active encoding is not
 *           QNN_QUANTIZATION_ENCODING_BW_SCALE_OFFSET.
 */
QNN_SYSTEM_API
Qnn_ErrorHandle_t QnnSystemDlc_getQuantParamsBwScaleOffsetScale(
    QnnSystemDlc_QuantParamsHandle_t quantParamsHandle, float* scale);

/**
 * @brief Returns the offset for QNN_QUANTIZATION_ENCODING_BW_SCALE_OFFSET encoding.
 *
 * @param[in]  quantParamsHandle  Handle obtained from QnnSystemDlc_getTensorInfoQuantParams.
 * @param[out] offset             Offset value.
 *
 * @return Error code
 *         - QNN_SUCCESS: Successfully retrieved offset.
 *         - QNN_SYSTEM_DLC_ERROR_INVALID_HANDLE: Invalid quantParamsHandle.
 *         - QNN_SYSTEM_DLC_ERROR_INVALID_ARGUMENT: offset is NULL, or active encoding is not
 *           QNN_QUANTIZATION_ENCODING_BW_SCALE_OFFSET.
 */
QNN_SYSTEM_API
Qnn_ErrorHandle_t QnnSystemDlc_getQuantParamsBwScaleOffsetOffset(
    QnnSystemDlc_QuantParamsHandle_t quantParamsHandle, int32_t* offset);

/**
 * @brief Returns the axis for QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET encoding.
 *
 * @param[in]  quantParamsHandle  Handle obtained from QnnSystemDlc_getTensorInfoQuantParams.
 * @param[out] axis               Axis value.
 *
 * @return Error code
 *         - QNN_SUCCESS: Successfully retrieved axis.
 *         - QNN_SYSTEM_DLC_ERROR_INVALID_HANDLE: Invalid quantParamsHandle.
 *         - QNN_SYSTEM_DLC_ERROR_INVALID_ARGUMENT: axis is NULL, or active encoding is not
 *           QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET.
 */
QNN_SYSTEM_API
Qnn_ErrorHandle_t QnnSystemDlc_getQuantParamsAxisScaleOffsetAxis(
    QnnSystemDlc_QuantParamsHandle_t quantParamsHandle, int32_t* axis);

/**
 * @brief Returns the number of scale/offset pairs for QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET.
 *
 * @param[in]  quantParamsHandle  Handle obtained from QnnSystemDlc_getTensorInfoQuantParams.
 * @param[out] numScaleOffsets    Number of scale/offset pairs.
 *
 * @return Error code
 *         - QNN_SUCCESS: Successfully retrieved count.
 *         - QNN_SYSTEM_DLC_ERROR_INVALID_HANDLE: Invalid quantParamsHandle.
 *         - QNN_SYSTEM_DLC_ERROR_INVALID_ARGUMENT: numScaleOffsets is NULL, or active encoding is
 *           not QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET.
 */
QNN_SYSTEM_API
Qnn_ErrorHandle_t QnnSystemDlc_getQuantParamsAxisScaleOffsetNumScaleOffsets(
    QnnSystemDlc_QuantParamsHandle_t quantParamsHandle, uint32_t* numScaleOffsets);

/**
 * @brief Returns the scale at a given index for QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET.
 *
 * @param[in]  quantParamsHandle  Handle obtained from QnnSystemDlc_getTensorInfoQuantParams.
 * @param[in]  index              Zero-based index (must be < numScaleOffsets).
 * @param[out] scale              Scale value at the given index.
 *
 * @return Error code
 *         - QNN_SUCCESS: Successfully retrieved scale.
 *         - QNN_SYSTEM_DLC_ERROR_INVALID_HANDLE: Invalid quantParamsHandle.
 *         - QNN_SYSTEM_DLC_ERROR_INVALID_ARGUMENT: scale is NULL, index out of range, or active
 *           encoding is not QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET.
 */
QNN_SYSTEM_API
Qnn_ErrorHandle_t QnnSystemDlc_getQuantParamsAxisScaleOffsetScaleAtIndex(
    QnnSystemDlc_QuantParamsHandle_t quantParamsHandle, uint32_t index, float* scale);

/**
 * @brief Returns the offset at a given index for QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET.
 *
 * @param[in]  quantParamsHandle  Handle obtained from QnnSystemDlc_getTensorInfoQuantParams.
 * @param[in]  index              Zero-based index (must be < numScaleOffsets).
 * @param[out] offset             Offset value at the given index.
 *
 * @return Error code
 *         - QNN_SUCCESS: Successfully retrieved offset.
 *         - QNN_SYSTEM_DLC_ERROR_INVALID_HANDLE: Invalid quantParamsHandle.
 *         - QNN_SYSTEM_DLC_ERROR_INVALID_ARGUMENT: offset is NULL, index out of range, or active
 *           encoding is not QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET.
 */
QNN_SYSTEM_API
Qnn_ErrorHandle_t QnnSystemDlc_getQuantParamsAxisScaleOffsetOffsetAtIndex(
    QnnSystemDlc_QuantParamsHandle_t quantParamsHandle, uint32_t index, int32_t* offset);

/**
 * @brief Returns the bitwidth for QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET encoding.
 *
 * @param[in]  quantParamsHandle  Handle obtained from QnnSystemDlc_getTensorInfoQuantParams.
 * @param[out] bitwidth           Bitwidth value.
 *
 * @return Error code
 *         - QNN_SUCCESS: Successfully retrieved bitwidth.
 *         - QNN_SYSTEM_DLC_ERROR_INVALID_HANDLE: Invalid quantParamsHandle.
 *         - QNN_SYSTEM_DLC_ERROR_INVALID_ARGUMENT: bitwidth is NULL, or active encoding is not
 *           QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET.
 */
QNN_SYSTEM_API
Qnn_ErrorHandle_t QnnSystemDlc_getQuantParamsBwAxisScaleOffsetBitwidth(
    QnnSystemDlc_QuantParamsHandle_t quantParamsHandle, uint32_t* bitwidth);

/**
 * @brief Returns the axis for QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET encoding.
 *
 * @param[in]  quantParamsHandle  Handle obtained from QnnSystemDlc_getTensorInfoQuantParams.
 * @param[out] axis               Axis value.
 *
 * @return Error code
 *         - QNN_SUCCESS: Successfully retrieved axis.
 *         - QNN_SYSTEM_DLC_ERROR_INVALID_HANDLE: Invalid quantParamsHandle.
 *         - QNN_SYSTEM_DLC_ERROR_INVALID_ARGUMENT: axis is NULL, or active encoding is not
 *           QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET.
 */
QNN_SYSTEM_API
Qnn_ErrorHandle_t QnnSystemDlc_getQuantParamsBwAxisScaleOffsetAxis(
    QnnSystemDlc_QuantParamsHandle_t quantParamsHandle, int32_t* axis);

/**
 * @brief Returns the number of elements for QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET.
 *
 * @param[in]  quantParamsHandle  Handle obtained from QnnSystemDlc_getTensorInfoQuantParams.
 * @param[out] numElements        Number of elements.
 *
 * @return Error code
 *         - QNN_SUCCESS: Successfully retrieved count.
 *         - QNN_SYSTEM_DLC_ERROR_INVALID_HANDLE: Invalid quantParamsHandle.
 *         - QNN_SYSTEM_DLC_ERROR_INVALID_ARGUMENT: numElements is NULL, or active encoding is not
 *           QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET.
 */
QNN_SYSTEM_API
Qnn_ErrorHandle_t QnnSystemDlc_getQuantParamsBwAxisScaleOffsetNumElements(
    QnnSystemDlc_QuantParamsHandle_t quantParamsHandle, uint32_t* numElements);

/**
 * @brief Returns the scale at a given index for QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET.
 *
 * @param[in]  quantParamsHandle  Handle obtained from QnnSystemDlc_getTensorInfoQuantParams.
 * @param[in]  index              Zero-based index (must be < numElements).
 * @param[out] scale              Scale value at the given index.
 *
 * @return Error code
 *         - QNN_SUCCESS: Successfully retrieved scale.
 *         - QNN_SYSTEM_DLC_ERROR_INVALID_HANDLE: Invalid quantParamsHandle.
 *         - QNN_SYSTEM_DLC_ERROR_INVALID_ARGUMENT: scale is NULL, index out of range, or active
 *           encoding is not QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET.
 */
QNN_SYSTEM_API
Qnn_ErrorHandle_t QnnSystemDlc_getQuantParamsBwAxisScaleOffsetScaleAtIndex(
    QnnSystemDlc_QuantParamsHandle_t quantParamsHandle, uint32_t index, float* scale);

/**
 * @brief Returns the offset at a given index for QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET.
 *
 * @param[in]  quantParamsHandle  Handle obtained from QnnSystemDlc_getTensorInfoQuantParams.
 * @param[in]  index              Zero-based index (must be < numElements).
 * @param[out] offset             Offset value at the given index.
 *
 * @return Error code
 *         - QNN_SUCCESS: Successfully retrieved offset.
 *         - QNN_SYSTEM_DLC_ERROR_INVALID_HANDLE: Invalid quantParamsHandle.
 *         - QNN_SYSTEM_DLC_ERROR_INVALID_ARGUMENT: offset is NULL, index out of range, or active
 *           encoding is not QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET.
 */
QNN_SYSTEM_API
Qnn_ErrorHandle_t QnnSystemDlc_getQuantParamsBwAxisScaleOffsetOffsetAtIndex(
    QnnSystemDlc_QuantParamsHandle_t quantParamsHandle, uint32_t index, int32_t* offset);

/**
 * @brief Returns context binary metadata for a prepared context info handle.
 *
 * Provides the same reflection as QnnSystemContext_getMetadata for a context binary
 * identified by its handle within the DLC.
 *
 * @param[in]  contextBinaryInfoHandle  Handle obtained from QnnSystemDlc_getContextBinaryInfoHandles.
 * @param[out] binaryInfo               Populated binary info struct (backend ID, graph list,
 *                                      tensor signatures, platform info, etc.).
 *                                      Memory owned by the DLC system library; valid until
 *                                      dlcHandle is freed.
 *
 * @return Error code
 *         - QNN_SUCCESS: Successfully populated binaryInfo.
 *         - QNN_SYSTEM_DLC_ERROR_INVALID_HANDLE: Invalid contextBinaryInfoHandle.
 *         - QNN_SYSTEM_DLC_ERROR_INVALID_ARGUMENT: binaryInfo is NULL.
 *         - QNN_SYSTEM_DLC_ERROR_MALFORMED_BINARY: Context binary cannot be parsed.
 *         - QNN_SYSTEM_DLC_ERROR_OPERATION_FAILED: Failed to obtain binary info.
 */
QNN_SYSTEM_API
Qnn_ErrorHandle_t QnnSystemDlc_getContextBinaryInfo(
    QnnSystemDlc_ContextBinaryInfoHandle_t contextBinaryInfoHandle,
    const QnnSystemContext_BinaryInfo_t** binaryInfo);

// ===========================================================================
// Handle-based getters for QnnSystemContext_BinaryInfo_t fields
// ===========================================================================

/**
 * @brief Get the backend ID from a context binary info handle.
 * @param[in]  contextBinaryInfoHandle  Handle from QnnSystemDlc_getContextBinaryInfoHandles.
 * @param[out] backendId                Backend identifier.
 */
QNN_SYSTEM_API
Qnn_ErrorHandle_t QnnSystemDlc_getContextBinaryInfoBackendId(
    QnnSystemDlc_ContextBinaryInfoHandle_t contextBinaryInfoHandle, uint32_t* backendId);

/**
 * @brief Get the build ID string from a context binary info handle.
 * @param[in]  contextBinaryInfoHandle  Handle from QnnSystemDlc_getContextBinaryInfoHandles.
 * @param[out] buildId                  Pointer to the null-terminated build ID string.
 *                                      Valid until the parent dlcHandle is freed.
 */
QNN_SYSTEM_API
Qnn_ErrorHandle_t QnnSystemDlc_getContextBinaryInfoBuildId(
    QnnSystemDlc_ContextBinaryInfoHandle_t contextBinaryInfoHandle, const char** buildId);

/**
 * @brief Get the QNN core API version from a context binary info handle.
 * @param[in]  contextBinaryInfoHandle  Handle from QnnSystemDlc_getContextBinaryInfoHandles.
 * @param[out] major  Major version component.
 * @param[out] minor  Minor version component.
 * @param[out] patch  Patch version component.
 */
QNN_SYSTEM_API
Qnn_ErrorHandle_t QnnSystemDlc_getContextBinaryInfoCoreApiVersion(
    QnnSystemDlc_ContextBinaryInfoHandle_t contextBinaryInfoHandle,
    uint32_t* major,
    uint32_t* minor,
    uint32_t* patch);

/**
 * @brief Get the backend API version from a context binary info handle.
 * @param[in]  contextBinaryInfoHandle  Handle from QnnSystemDlc_getContextBinaryInfoHandles.
 * @param[out] major  Major version component.
 * @param[out] minor  Minor version component.
 * @param[out] patch  Patch version component.
 */
QNN_SYSTEM_API
Qnn_ErrorHandle_t QnnSystemDlc_getContextBinaryInfoBackendApiVersion(
    QnnSystemDlc_ContextBinaryInfoHandle_t contextBinaryInfoHandle,
    uint32_t* major,
    uint32_t* minor,
    uint32_t* patch);

/**
 * @brief Get the SOC version string from a context binary info handle.
 * @param[in]  contextBinaryInfoHandle  Handle from QnnSystemDlc_getContextBinaryInfoHandles.
 * @param[out] socVersion               Pointer to the null-terminated SOC version string.
 *                                      Valid until the parent dlcHandle is freed.
 */
QNN_SYSTEM_API
Qnn_ErrorHandle_t QnnSystemDlc_getContextBinaryInfoSocVersion(
    QnnSystemDlc_ContextBinaryInfoHandle_t contextBinaryInfoHandle, const char** socVersion);

/**
 * @brief Get the hardware info blob version from a context binary info handle.
 *        Returns QNN_SYSTEM_DLC_ERROR_UNSUPPORTED_FEATURE for V3 binaries (field absent).
 * @param[in]  contextBinaryInfoHandle  Handle from QnnSystemDlc_getContextBinaryInfoHandles.
 * @param[out] major  Major version component.
 * @param[out] minor  Minor version component.
 * @param[out] patch  Patch version component.
 */
QNN_SYSTEM_API
Qnn_ErrorHandle_t QnnSystemDlc_getContextBinaryInfoHwInfoBlobVersion(
    QnnSystemDlc_ContextBinaryInfoHandle_t contextBinaryInfoHandle,
    uint32_t* major,
    uint32_t* minor,
    uint32_t* patch);

/**
 * @brief Get the context blob version from a context binary info handle.
 * @param[in]  contextBinaryInfoHandle  Handle from QnnSystemDlc_getContextBinaryInfoHandles.
 * @param[out] major  Major version component.
 * @param[out] minor  Minor version component.
 * @param[out] patch  Patch version component.
 */
QNN_SYSTEM_API
Qnn_ErrorHandle_t QnnSystemDlc_getContextBinaryInfoContextBlobVersion(
    QnnSystemDlc_ContextBinaryInfoHandle_t contextBinaryInfoHandle,
    uint32_t* major,
    uint32_t* minor,
    uint32_t* patch);

/**
 * @brief Get the hardware info blob size from a context binary info handle.
 *        Returns QNN_SYSTEM_DLC_ERROR_UNSUPPORTED_FEATURE for V3 binaries (field absent).
 * @param[in]  contextBinaryInfoHandle  Handle from QnnSystemDlc_getContextBinaryInfoHandles.
 * @param[out] hwInfoBlobSize           Size in bytes.
 */
QNN_SYSTEM_API
Qnn_ErrorHandle_t QnnSystemDlc_getContextBinaryInfoHwInfoBlobSize(
    QnnSystemDlc_ContextBinaryInfoHandle_t contextBinaryInfoHandle, uint32_t* hwInfoBlobSize);

/**
 * @brief Get the hardware info blob pointer from a context binary info handle.
 *        Returns QNN_SYSTEM_DLC_ERROR_UNSUPPORTED_FEATURE for V3 binaries (field absent).
 * @param[in]  contextBinaryInfoHandle  Handle from QnnSystemDlc_getContextBinaryInfoHandles.
 * @param[out] hwInfoBlob               Pointer to the hardware info blob.
 *                                      Valid until the parent dlcHandle is freed.
 */
QNN_SYSTEM_API
Qnn_ErrorHandle_t QnnSystemDlc_getContextBinaryInfoHwInfoBlob(
    QnnSystemDlc_ContextBinaryInfoHandle_t contextBinaryInfoHandle, void** hwInfoBlob);

/**
 * @brief Get the context blob size from a context binary info handle.
 * @param[in]  contextBinaryInfoHandle  Handle from QnnSystemDlc_getContextBinaryInfoHandles.
 * @param[out] contextBlobSize          Size in bytes.
 */
QNN_SYSTEM_API
Qnn_ErrorHandle_t QnnSystemDlc_getContextBinaryInfoContextBlobSize(
    QnnSystemDlc_ContextBinaryInfoHandle_t contextBinaryInfoHandle, uint64_t* contextBlobSize);

/**
 * @brief Get the number of context-level tensors from a context binary info handle.
 * @param[in]  contextBinaryInfoHandle  Handle from QnnSystemDlc_getContextBinaryInfoHandles.
 * @param[out] numContextTensors        Number of context tensors.
 */
QNN_SYSTEM_API
Qnn_ErrorHandle_t QnnSystemDlc_getContextBinaryInfoNumContextTensors(
    QnnSystemDlc_ContextBinaryInfoHandle_t contextBinaryInfoHandle, uint32_t* numContextTensors);

/**
 * @brief Get a tensor info handle for a context-level tensor by index.
 *        The same index always returns the same handle (cached internally).
 * @param[in]  contextBinaryInfoHandle  Handle from QnnSystemDlc_getContextBinaryInfoHandles.
 * @param[in]  index                   Zero-based index; must be < numContextTensors.
 * @param[out] tensorInfoHandle        Populated with an opaque handle to the tensor info.
 */
QNN_SYSTEM_API
Qnn_ErrorHandle_t QnnSystemDlc_getContextBinaryInfoContextTensorInfo(
    QnnSystemDlc_ContextBinaryInfoHandle_t contextBinaryInfoHandle,
    uint32_t index,
    QnnSystemDlc_TensorInfoHandle_t* tensorInfoHandle);

/**
 * @brief Get the number of graphs from a context binary info handle.
 * @param[in]  contextBinaryInfoHandle  Handle from QnnSystemDlc_getContextBinaryInfoHandles.
 * @param[out] numGraphs                Number of graphs.
 */
QNN_SYSTEM_API
Qnn_ErrorHandle_t QnnSystemDlc_getContextBinaryInfoNumGraphs(
    QnnSystemDlc_ContextBinaryInfoHandle_t contextBinaryInfoHandle, uint32_t* numGraphs);

/**
 * @brief Get a context graph info handle by index.
 *        The same index always returns the same handle (cached internally).
 * @param[in]  contextBinaryInfoHandle   Handle from QnnSystemDlc_getContextBinaryInfoHandles.
 * @param[in]  index                    Zero-based index; must be < numGraphs.
 * @param[out] contextGraphInfoHandle   Populated with an opaque handle to the graph info.
 */
QNN_SYSTEM_API
Qnn_ErrorHandle_t QnnSystemDlc_getContextGraphInfoHandle(
    QnnSystemDlc_ContextBinaryInfoHandle_t contextBinaryInfoHandle,
    uint32_t index,
    QnnSystemDlc_ContextGraphInfoHandle_t* contextGraphInfoHandle);

/**
 * @brief Get the SOC model integer from a context binary info handle.
 *        Returns QNN_SYSTEM_DLC_ERROR_UNSUPPORTED_FEATURE for V1/V2 binaries (field absent).
 * @param[in]  contextBinaryInfoHandle  Handle from QnnSystemDlc_getContextBinaryInfoHandles.
 * @param[out] socModel                 Integer SoC model identifier.
 */
QNN_SYSTEM_API
Qnn_ErrorHandle_t QnnSystemDlc_getContextBinaryInfoSocModel(
    QnnSystemDlc_ContextBinaryInfoHandle_t contextBinaryInfoHandle, uint32_t* socModel);

/**
 * @brief Get the context metadata size from a context binary info handle.
 *        Returns QNN_SYSTEM_DLC_ERROR_UNSUPPORTED_FEATURE for V1/V2 binaries (field absent).
 * @param[in]  contextBinaryInfoHandle  Handle from QnnSystemDlc_getContextBinaryInfoHandles.
 * @param[out] contextMetadataSize      Size in bytes.
 */
QNN_SYSTEM_API
Qnn_ErrorHandle_t QnnSystemDlc_getContextBinaryInfoContextMetadataSize(
    QnnSystemDlc_ContextBinaryInfoHandle_t contextBinaryInfoHandle, uint32_t* contextMetadataSize);

/**
 * @brief Get the context metadata pointer from a context binary info handle.
 *        Returns QNN_SYSTEM_DLC_ERROR_UNSUPPORTED_FEATURE for V1/V2 binaries (field absent).
 * @param[in]  contextBinaryInfoHandle  Handle from QnnSystemDlc_getContextBinaryInfoHandles.
 * @param[out] contextMetadata          Pointer to the context metadata blob.
 *                                      Valid until the parent dlcHandle is freed.
 */
QNN_SYSTEM_API
Qnn_ErrorHandle_t QnnSystemDlc_getContextBinaryInfoContextMetadata(
    QnnSystemDlc_ContextBinaryInfoHandle_t contextBinaryInfoHandle, void** contextMetadata);

// ===========================================================================
// Handle-based getters for QnnSystemContext_GraphInfo_t fields
// (context binary graph entries, accessed via QnnSystemDlc_ContextGraphInfoHandle_t)
// ===========================================================================

/**
 * @brief Get the graph name from a context graph info handle.
 * @param[in]  contextGraphInfoHandle  Handle from QnnSystemDlc_getContextGraphInfoHandle.
 * @param[out] graphName               Pointer to the null-terminated graph name string.
 *                                     Valid until the parent dlcHandle is freed.
 */
QNN_SYSTEM_API
Qnn_ErrorHandle_t QnnSystemDlc_getContextGraphInfoGraphName(
    QnnSystemDlc_ContextGraphInfoHandle_t contextGraphInfoHandle, const char** graphName);

/**
 * @brief Get the number of graph input tensors from a context graph info handle.
 * @param[in]  contextGraphInfoHandle  Handle from QnnSystemDlc_getContextGraphInfoHandle.
 * @param[out] numInputTensors         Number of input tensors.
 */
QNN_SYSTEM_API
Qnn_ErrorHandle_t QnnSystemDlc_getContextGraphInfoNumInputTensors(
    QnnSystemDlc_ContextGraphInfoHandle_t contextGraphInfoHandle, uint32_t* numInputTensors);

/**
 * @brief Get the number of graph output tensors from a context graph info handle.
 * @param[in]  contextGraphInfoHandle  Handle from QnnSystemDlc_getContextGraphInfoHandle.
 * @param[out] numOutputTensors        Number of output tensors.
 */
QNN_SYSTEM_API
Qnn_ErrorHandle_t QnnSystemDlc_getContextGraphInfoNumOutputTensors(
    QnnSystemDlc_ContextGraphInfoHandle_t contextGraphInfoHandle, uint32_t* numOutputTensors);

/**
 * @brief Get the number of updatable tensors from a context graph info handle.
 *        Returns QNN_SYSTEM_DLC_ERROR_UNSUPPORTED_FEATURE for V1 graph info (field absent).
 * @param[in]  contextGraphInfoHandle    Handle from QnnSystemDlc_getContextGraphInfoHandle.
 * @param[out] numUpdateableTensors      Number of updatable tensors.
 */
QNN_SYSTEM_API
Qnn_ErrorHandle_t QnnSystemDlc_getContextGraphInfoNumUpdateableTensors(
    QnnSystemDlc_ContextGraphInfoHandle_t contextGraphInfoHandle, uint32_t* numUpdateableTensors);

/**
 * @brief Get the graph blob info size from a context graph info handle.
 *        Returns QNN_SYSTEM_DLC_ERROR_UNSUPPORTED_FEATURE for V1/V2 graph info (field absent).
 * @param[in]  contextGraphInfoHandle  Handle from QnnSystemDlc_getContextGraphInfoHandle.
 * @param[out] graphBlobInfoSize       Size in bytes.
 */
QNN_SYSTEM_API
Qnn_ErrorHandle_t QnnSystemDlc_getContextGraphInfoGraphBlobInfoSize(
    QnnSystemDlc_ContextGraphInfoHandle_t contextGraphInfoHandle, uint32_t* graphBlobInfoSize);

/**
 * @brief Get the graph blob info pointer from a context graph info handle.
 *        Returns QNN_SYSTEM_DLC_ERROR_UNSUPPORTED_FEATURE for V1/V2 graph info (field absent).
 * @param[in]  contextGraphInfoHandle  Handle from QnnSystemDlc_getContextGraphInfoHandle.
 * @param[out] graphBlobInfo           Pointer to the graph info blob.
 *                                     Valid until the parent dlcHandle is freed.
 */
QNN_SYSTEM_API
Qnn_ErrorHandle_t QnnSystemDlc_getContextGraphInfoGraphBlobInfo(
    QnnSystemDlc_ContextGraphInfoHandle_t contextGraphInfoHandle, void** graphBlobInfo);

/**
 * @brief Get the start op index from a context graph info handle.
 *        Returns QNN_SYSTEM_DLC_ERROR_UNSUPPORTED_FEATURE for V1/V2 graph info (field absent).
 * @param[in]  contextGraphInfoHandle  Handle from QnnSystemDlc_getContextGraphInfoHandle.
 * @param[out] startOpIndex            Start op index.
 */
QNN_SYSTEM_API
Qnn_ErrorHandle_t QnnSystemDlc_getContextGraphInfoStartOpIndex(
    QnnSystemDlc_ContextGraphInfoHandle_t contextGraphInfoHandle, uint32_t* startOpIndex);

/**
 * @brief Get the end op index from a context graph info handle.
 *        Returns QNN_SYSTEM_DLC_ERROR_UNSUPPORTED_FEATURE for V1/V2 graph info (field absent).
 * @param[in]  contextGraphInfoHandle  Handle from QnnSystemDlc_getContextGraphInfoHandle.
 * @param[out] endOpIndex              End op index.
 */
QNN_SYSTEM_API
Qnn_ErrorHandle_t QnnSystemDlc_getContextGraphInfoEndOpIndex(
    QnnSystemDlc_ContextGraphInfoHandle_t contextGraphInfoHandle, uint32_t* endOpIndex);

/**
 * @brief Get a tensor info handle for a graph input tensor by index.
 *        The same index always returns the same handle (cached internally).
 * @param[in]  contextGraphInfoHandle  Handle from QnnSystemDlc_getContextGraphInfoHandle.
 * @param[in]  index                   Zero-based index; must be < numInputTensors.
 * @param[out] tensorInfoHandle        Populated with an opaque handle to the tensor info.
 */
QNN_SYSTEM_API
Qnn_ErrorHandle_t QnnSystemDlc_getContextGraphInfoInputTensorInfo(
    QnnSystemDlc_ContextGraphInfoHandle_t contextGraphInfoHandle,
    uint32_t index,
    QnnSystemDlc_TensorInfoHandle_t* tensorInfoHandle);

/**
 * @brief Get a tensor info handle for a graph output tensor by index.
 *        The same index always returns the same handle (cached internally).
 * @param[in]  contextGraphInfoHandle  Handle from QnnSystemDlc_getContextGraphInfoHandle.
 * @param[in]  index                   Zero-based index; must be < numOutputTensors.
 * @param[out] tensorInfoHandle        Populated with an opaque handle to the tensor info.
 */
QNN_SYSTEM_API
Qnn_ErrorHandle_t QnnSystemDlc_getContextGraphInfoOutputTensorInfo(
    QnnSystemDlc_ContextGraphInfoHandle_t contextGraphInfoHandle,
    uint32_t index,
    QnnSystemDlc_TensorInfoHandle_t* tensorInfoHandle);

/**
 * @brief Get a tensor info handle for an updatable tensor by index.
 *        Returns QNN_SYSTEM_DLC_ERROR_UNSUPPORTED_FEATURE for V1 graph info.
 *        The same index always returns the same handle (cached internally).
 * @param[in]  contextGraphInfoHandle  Handle from QnnSystemDlc_getContextGraphInfoHandle.
 * @param[in]  index                   Zero-based index; must be < numUpdateableTensors.
 * @param[out] tensorInfoHandle        Populated with an opaque handle to the tensor info.
 */
QNN_SYSTEM_API
Qnn_ErrorHandle_t QnnSystemDlc_getContextGraphInfoUpdateableTensorInfo(
    QnnSystemDlc_ContextGraphInfoHandle_t contextGraphInfoHandle,
    uint32_t index,
    QnnSystemDlc_TensorInfoHandle_t* tensorInfoHandle);

/**
 * @brief A function to free the instance of the System DLC object
 *        This API clears any intermediate memory allocated and associated
 *        with a valid handle
 *
 * @param[in] dlcHandle Handle to the System DLC object
 *
 * @return Error code
 *         - QNN_SUCCESS: Successfully freed instance of System DLC
 *         - QNN_SYSTEM_DLC_ERROR_INVALID_HANDLE: Invalid System DLC handle to free
 *         - QNN_SYSTEM_DLC_ERROR_UNSUPPORTED_FEATURE: not supported
 */
QNN_SYSTEM_API
Qnn_ErrorHandle_t QnnSystemDlc_free(QnnSystemDlc_Handle_t dlcHandle);

/**
 * @brief A function to save the DLC __dlcHandle__ to the location path.
 *
 * @param[in] dlcHandle the handle to which the record is associated
 * @param[in] path path to save the DLC handle
 */
QNN_SYSTEM_API
Qnn_ErrorHandle_t QnnSystemDlc_save(QnnSystemDlc_Handle_t dlcHandle, const char* path);

/**
 * @brief A function to get the size of memory needed to hold the DLC in binary form.
 *        Must be called before QnnSystemDlc_getBinary() to determine the required buffer size.
 *
 * @param[in]  dlcHandle        A handle to the DLC instance.
 * @param[out] binaryBufferSize The number of bytes the client must allocate to hold the
 *                              serialized DLC binary.
 *
 * @return Error code
 *         - QNN_SUCCESS: Successfully retrieved the binary size
 *         - QNN_SYSTEM_DLC_ERROR_INVALID_HANDLE: dlcHandle is not a valid handle
 *         - QNN_SYSTEM_DLC_ERROR_INVALID_ARGUMENT: binaryBufferSize is NULL
 *         - QNN_SYSTEM_DLC_ERROR_OPERATION_FAILED: failed to compute binary size
 *         - QNN_SYSTEM_DLC_ERROR_UNSUPPORTED_FEATURE: feature not supported
 */
QNN_SYSTEM_API
Qnn_ErrorHandle_t QnnSystemDlc_getBinarySize(QnnSystemDlc_Handle_t dlcHandle,
                                             Qnn_SystemDlcBinarySize_t* binaryBufferSize);

/**
 * @brief A function to serialize the DLC into a client-provided buffer.
 *        The client must first call QnnSystemDlc_getBinarySize() to determine the required
 *        buffer size and allocate sufficient memory before calling this function.
 *
 * @param[in]  dlcHandle         A handle to the DLC instance.
 * @param[in]  binaryBuffer      Pointer to the client-allocated buffer to write the DLC into.
 * @param[in]  binaryBufferSize  Size of binaryBuffer in bytes.
 * @param[out] writtenBufferSize Number of bytes actually written into binaryBuffer.
 *
 * @return Error code
 *         - QNN_SUCCESS: Successfully serialized the DLC into binaryBuffer
 *         - QNN_SYSTEM_DLC_ERROR_INVALID_HANDLE: dlcHandle is not a valid handle
 *         - QNN_SYSTEM_DLC_ERROR_INVALID_ARGUMENT: binaryBuffer or writtenBufferSize is NULL
 *         - QNN_SYSTEM_DLC_ERROR_OPERATION_FAILED: failed to serialize the DLC
 *         - QNN_SYSTEM_DLC_ERROR_UNSUPPORTED_FEATURE: feature not supported
 */
QNN_SYSTEM_API
Qnn_ErrorHandle_t QnnSystemDlc_getBinary(QnnSystemDlc_Handle_t dlcHandle,
                                         uint8_t* binaryBuffer,
                                         Qnn_SystemDlcBinarySize_t binaryBufferSize,
                                         Qnn_SystemDlcBinarySize_t* writtenBufferSize);

/**
 * @brief A function to create an instance of the empty DLC with an optional destination directory
 *        hint
 *
 * @param[in] logger a log handle produced from QnnSystemLog_create(). Can be NULL
 * @param[in] destinationDirectoryHint a hint for the working directory used for temporary
 *            archive files. Can be NULL or empty string, in which case a default working directory
 *            is used.
 * @param[out] dlcHandle A handle to the created instance of a systemDlc entity
 *
 * @return Error code
 *         - QNN_SUCCESS: Successfully created a systemDlc entity
 *         - QNN_SYSTEM_DLC_ERROR_INVALID_ARGUMENT: dlcHandle is NULL
 *         - QNN_COMMON_ERROR_MEM_ALLOC: Error encountered in allocating memory for
 *           systemDlc instance
 *         - QNN_SYSTEM_DLC_ERROR_UNSUPPORTED_FEATURE: system Dlc features not supported
 */
QNN_SYSTEM_API
Qnn_ErrorHandle_t QnnSystemDlc_createWithDestinationDir(Qnn_LogHandle_t logger,
                                                        const char* destinationDirectoryHint,
                                                        QnnSystemDlc_Handle_t* dlcHandle);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // QNN_SYSTEM_DLC_H
