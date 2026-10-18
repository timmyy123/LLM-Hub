//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

/**
 * @file
 * @brief   QNN System Common API component
 *
 *          A header which contains common types shared by QNN system components.
 *          This simplifies the cross-inclusion of headers.
 */

#ifndef QNN_SYSTEM_COMMON_H
#define QNN_SYSTEM_COMMON_H

#include "QnnCommon.h"

#ifdef __cplusplus
extern "C" {
#endif

//=============================================================================
// Error Codes
//=============================================================================

#define QNN_SYSTEM_COMMON_MIN_ERROR  QNN_MIN_ERROR_SYSTEM
#define QNN_SYSTEM_COMMON_MAX_ERROR  QNN_MIN_ERROR_SYSTEM + 999
#define QNN_SYSTEM_PROFILE_MIN_ERROR QNN_MIN_ERROR_SYSTEM + 1000
#define QNN_SYSTEM_PROFILE_MAX_ERROR QNN_MIN_ERROR_SYSTEM + 1999

/**
 * @brief QNN System Profile API result / error codes.
 */
typedef enum {
  /// Qnn System success
  QNN_SYSTEM_COMMON_NO_ERROR = QNN_SUCCESS,
  /// There is an API component that is not supported yet.
  QNN_SYSTEM_COMMON_ERROR_UNSUPPORTED_FEATURE = QNN_COMMON_ERROR_NOT_SUPPORTED,
  /// QNN System invalid handle
  QNN_SYSTEM_COMMON_ERROR_INVALID_HANDLE = QNN_SYSTEM_COMMON_MIN_ERROR + 0,
  /// One or more arguments to a System API is/are NULL/invalid.
  QNN_SYSTEM_COMMON_ERROR_INVALID_ARGUMENT = QNN_SYSTEM_COMMON_MIN_ERROR + 1
} QnnSystemCommon_Error_t;

//=============================================================================
// Macros
//=============================================================================

// libQnnSystem.so system interface provider name
#define QNN_SYSTEM_INTERFACE_PROVIDER_NAME "SYSTEM_QTI_AISW"

// Macro controlling visibility of QNN_SYSTEM API
#ifndef QNN_SYSTEM_API
#define QNN_SYSTEM_API
#endif

// Provide values to use for API version.
#define QNN_SYSTEM_API_VERSION_MAJOR 1
#define QNN_SYSTEM_API_VERSION_MINOR 14
#define QNN_SYSTEM_API_VERSION_PATCH 0

// Error code space assigned to system API components
#define QNN_SYSTEM_CONTEXT_MIN_ERROR QNN_MIN_ERROR_SYSTEM
#define QNN_SYSTEM_CONTEXT_MAX_ERROR (QNN_SYSTEM_CONTEXT_MIN_ERROR + 999)

//=============================================================================
// Data Types
//=============================================================================

/// Size type for DLC binary buffers, in bytes.
typedef uint64_t Qnn_SystemDlcBinarySize_t;

//=============================================================================
// Public Functions
//=============================================================================

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // QNN_SYSTEM_COMMON_H
