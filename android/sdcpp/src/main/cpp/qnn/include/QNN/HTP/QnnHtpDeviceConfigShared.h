//=============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================

/** @file
 *  @brief QNN HTP device configuration types shared between on-chip and MCP builds.
 *
 *  Included by both QnnHtpDevice.h and QnnHtpMcpDevice.h.
 */

#pragma once

#include "QnnCommon.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  QNN_HTP_DEVICE_ARCH_NONE    = 0,
  QNN_HTP_DEVICE_ARCH_V68     = 68,
  QNN_HTP_DEVICE_ARCH_V69     = 69,
  QNN_HTP_DEVICE_ARCH_V73     = 73,
  QNN_HTP_DEVICE_ARCH_V75     = 75,
  QNN_HTP_DEVICE_ARCH_V79     = 79,
  QNN_HTP_DEVICE_ARCH_V81     = 81,
  QNN_HTP_DEVICE_ARCH_V85     = 85,
  QNN_HTP_DEVICE_ARCH_V89     = 89,
  QNN_HTP_DEVICE_ARCH_UNKNOWN = 0x7fffffff
} QnnHtpDevice_Arch_t;

typedef struct {
  uint32_t deviceId;
  QnnHtpDevice_Arch_t arch;
} QnnHtpDevice_Minimum_Arch_t;

typedef struct {
  uint32_t deviceId;
  bool useSignedProcessDomain;
} QnnHtpDevice_UseSignedProcessDomain_t;

/** @deprecated SecurePD for model protection is no longer supported. Use of
 * QnnHtpDevice_UseSecureProcessDomain_t via QNN_HTP_DEVICE_CONFIG_OPTION_SECUREPD will result in
 * QNN_COMMON_ERROR_NOT_SUPPORTED at runtime. */
typedef struct {
  uint32_t deviceId;
  bool useSecureProcessDomain;
} QnnHtpDevice_UseSecureProcessDomain_t;

typedef struct {
  uint32_t deviceId;
  bool enableAsyncExecute;
} QnnHtpDevice_AsyncExecute_t;

typedef struct {
  uint32_t deviceId;
  bool useDynamicPartitioning;
  // Opaque handle to the RSM (Resource Sharing Manager) context associated with dynamic partitioning
  void* rsmHandle;
} QnnHtpDevice_UseDynamicPartitioning_t;

typedef enum {
  QNN_HTP_DEVICE_CONFIG_OPTION_SOC      = 0,
  QNN_HTP_DEVICE_CONFIG_OPTION_ARCH     = 1,
  QNN_HTP_DEVICE_CONFIG_OPTION_SIGNEDPD = 2,
  QNN_HTP_DEVICE_CONFIG_OPTION_SECUREPD =
      3,  // Deprecated: SecurePD for model protection is no longer supported
  QNN_HTP_DEVICE_CONFIG_OPTION_ASYNC_EXECUTE = 4,
  QNN_HTP_DEVICE_CONFIG_OPTION_DYNAMIC_PARTITIONING = 5,
  QNN_HTP_DEVICE_CONFIG_OPTION_APP_UID       = 6,
  QNN_HTP_DEVICE_CONFIG_OPTION_HTP_EXTENSION = 0x7fff0000,
  QNN_HTP_DEVICE_CONFIG_OPTION_UNKNOWN       = 0x7fffffff
} QnnHtpDevice_ConfigOption_t;

typedef struct {
  QnnHtpDevice_ConfigOption_t option;
  union UNNAMED {
    uint32_t socModel;
    QnnHtpDevice_Minimum_Arch_t arch;
    QnnHtpDevice_UseSignedProcessDomain_t useSignedProcessDomain;
    QnnHtpDevice_UseSecureProcessDomain_t
        useSecureProcessDomain;  // Deprecated: SecurePD for model protection is no longer supported
    QnnHtpDevice_AsyncExecute_t asyncExecute;
    QnnHtpDevice_UseDynamicPartitioning_t useDynamicPartitioning;
    uint32_t appUid;
  };
} QnnHtpDevice_CustomConfig_t;

#ifdef __cplusplus
}  // extern "C"
#endif
