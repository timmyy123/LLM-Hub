//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All rights reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

/**
 *  @file
 *  @brief QNN HTP component System Context API.
 *
 *         The interfaces in this file work with the top level QNN
 *         API and supplements QnnSystemContext.h for HTP backend
 */

#ifndef QNN_HTP_SYSTEM_CONTEXT_H
#define QNN_HTP_SYSTEM_CONTEXT_H

#ifdef __cplusplus
extern "C" {
#endif

//=============================================================================
// Macros
//=============================================================================
typedef enum {
  // Following version with hwInfoBlobVersion as:
  //   - Major 0, Minor: 0, Patch: 1
  QNN_SYSTEM_CONTEXT_HTP_HW_INFO_BLOB_VERSION_V1 = 0x01,
  // Unused, present to ensure 32 bits.
  QNN_SYSTEM_CONTEXT_HTP_HW_INFO_BLOB_UNDEFINED = 0x7FFFFFFF
} QnnHtpSystemContext_HwInfoBlobVersion_t;

// This struct is gets populated within a binary blob as part of hwInfoBlob in
// QnnSystemContext_BinaryInfoV#_t struct in QnnSystemContext.h
typedef struct QnnHtpSystemContext_HwBlobInfoV1 {
  // This value represents the index of the list of graphs registered
  // to this context as specified in QnnSystemContext_GraphInfo_t*
  uint32_t graphListIndex;
  // Stores the spill-fill buffer size used by each of the graphs
  uint64_t spillFillBufferSize;
} QnnHtpSystemContext_HwBlobInfoV1_t;

typedef struct {
  QnnHtpSystemContext_HwInfoBlobVersion_t version;
  union UNNAMED {
    QnnHtpSystemContext_HwBlobInfoV1_t contextBinaryHwInfoBlobV1_t;
  };
} QnnHtpSystemContext_HwBlobInfo_t;

typedef enum {
  // Following version with GraphInfoBlobVersion as:
  //   - Major 0, Minor: 0, Patch: 1
  QNN_SYSTEM_CONTEXT_HTP_GRAPH_INFO_BLOB_VERSION_V1 = 0x01,
  QNN_SYSTEM_CONTEXT_HTP_GRAPH_INFO_BLOB_VERSION_V2 = 0x02,
  // Unused, present to ensure 32 bits.
  QNN_SYSTEM_CONTEXT_HTP_GRAPH_INFO_BLOB_UNDEFINED = 0x7FFFFFFF
} QnnHtpSystemContext_GraphInfoBlobVersion_t;

// This struct is gets populated within a binary blob as part of GraphInfoBlob in
// QnnSystemContext_BinaryInfoV1_t struct in QnnSystemContext.h
typedef struct {
  // Stores the spill-fill buffer size used by each of the graphs
  uint64_t spillFillBufferSize;
  // HTP vtcm size (MB)
  uint32_t vtcmSize;
  // Optimization level
  uint32_t optimizationLevel;
  // Htp Dlbc
  uint8_t htpDlbc;
  // Number of HVX Threads to reserve;
  uint64_t numHvxThreads;
} QnnHtpSystemContext_GraphBlobInfoV1_t;

// This struct gets populated within a binary blob as part of GraphInfoBlob in
// QnnSystemContext_BinaryInfoV2_t struct in QnnSystemContext.h
/*
Note: This chart is for illustrative purposes only.
+-----------------------------+--------------------------+
|        256G (Far Mem)       |                          |
|                             | Shared (name)     : 6G   |
|                             +--------------------------+
|                             | Non-Shared (name) : 1M   |
|                             +--------------------------+
|                             | I/O                      |
|                             | K Cache (name)    : 100M |
|                             | V Cache (name)    : 100M |
|                             +--------------------------+
|                             | Free Memory       : 100M |
|                             +--------------------------+
|                             |                          |
|                             | (Far Memory starts at 4G)|
+-----------------------------+--------------------------+
|         3.5G (Near Mem)     |                          |
|                             | Shared (name)     : 1G   |
|                             +--------------------------+
|                             | Non-Shared (Const): 100K |
|                             +--------------------------+
|                             | Op Data           : 100K |
|                             |                          |
+-----------------------------+--------------------------+

                Total Memory Used: 6.701G
*/
typedef struct {
  QnnHtpSystemContext_GraphInfoBlobVersion_t version;
  // Stores the nativeK channel tile size used by each of the graphs (bytes)
  uint64_t nativeKChannelSize;
  // Stores the nativeV channel tile size used by each of the graphs (bytes)
  uint64_t nativeVChannelSize;
  // The field name IsSafeShareIO indicates if it is safe to share the
  // buffer between inputs and outputs. 1: True, 0: False
  // Client is responsible for ensuring no clash between input and output
  // when flag is set.
  uint32_t isSafeShareIO;
  // Stores graph input/output tensors size (bytes)
  uint64_t ioTensorSize;
  // Stores opdata memory/meta data size associated with ops that will be executed,
  // inlcuding op data like runlists (bytes)
  uint64_t opDataSize;
  // Stores size of const data in the graph (bytes)
  uint64_t constSize;
  // Stores size of DDR-tensor (bytes)
  uint64_t ddrTensorSize;
  // Stores shared weights size (bytes)
  uint64_t sharedWeightsSize;

} QnnHtpSystemContext_GraphBlobInfoV2_t;

typedef struct {
  QnnHtpSystemContext_GraphInfoBlobVersion_t version;
  union UNNAMED {
    QnnHtpSystemContext_GraphBlobInfoV1_t contextBinaryGraphBlobInfoV1;
  };
} QnnHtpSystemContext_GraphBlobInfo_t;

typedef enum {
  // Following version with ContextInfoBlobVersion as:
  //   - Major 0, Minor: 0, Patch: 1
  QNN_SYSTEM_CONTEXT_HTP_CONTEXT_INFO_BLOB_VERSION_V1 = 0x01,
  // Unused, present to ensure 32 bits.
  QNN_SYSTEM_CONTEXT_HTP_CONTEXT_INFO_BLOB_UNDEFINED = 0x7FFFFFFF
} QnnHtpSystemContext_ContextInfoBlobVersion_t;

typedef struct {
  /// An integer representation of SocUtility::DspArch
  uint32_t dspArch;
} QnnHtpSystemContext_ContextBlobInfoV1_t;

typedef struct {
  QnnHtpSystemContext_ContextInfoBlobVersion_t version;
  union UNNAMED {
  QnnHtpSystemContext_ContextBlobInfoV1_t contextBinaryContextBlobInfoV1;
  };
} QnnHtpSystemContext_ContextBlobInfo_t;


//=============================================================================
// Data Types
//=============================================================================

//=============================================================================
// Public Functions
//=============================================================================

//=============================================================================
// Implementation Definition
//=============================================================================

// clang-format on
#ifdef __cplusplus
}  // extern "C"
#endif

#endif