//=============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================

/** @file
 *  @brief QNN HTP Device components
 *
 *  This file defines structures and supplements QnnDevice.h for QNN HTP device
 */

#pragma once

#include "QnnCommon.h"
#include "QnnDevice.h"
#include "QnnHtpDeviceConfigShared.h"
#include "QnnHtpPerfInfrastructure.h"
#include "QnnTypes.h"
#ifdef __cplusplus
extern "C" {
#endif

// For deviceType in QnnDevice_HardwareDeviceInfoV1_t
typedef enum {
  QNN_HTP_DEVICE_TYPE_ON_CHIP = 0,  // HTP cores are inside SoC
  QNN_HTP_DEVICE_TYPE_OFF_CHIP = 1,
  QNN_HTP_DEVICE_TYPE_UNKNOWN = 0x7fffffff
} QnnHtpDevice_DeviceType_t;

/**
 * @brief QNN HTP Device core type
 * This enumeration provides information about the core type inside the SOC.
 *
 * For online operation, the caller should retrieve this information from
 * `QnnDevice_getPlatformInfo`. For offline operation, the caller needs to create a
 * `QnnDevice_CoreInfo_t` with the correct core type, and then use it to create the
 * `QnnDevice_PlatformInfo_t`.
 */
typedef enum {
  QNN_HTP_CORE_TYPE_NSP   = 0,
  QNN_HTP_CORE_TYPE_HPASS = 1,

  // supported coreType are < QNN_CORE_TYPE_MAX
  QNN_HTP_CORE_TYPE_MAX,
  QNN_HTP_CORE_TYPE_UNKNOWN = 0x7fffffff
} QnnHtpDevice_CoreType_t;

/**
 * This structure provides info about the NSP device inside SoC
 * For online operation, caller should get these info from QnnDevice_getPlatformInfo
 * For offline operation, caller need to create this structure and filling the correct information
 * for QnnDevice_create
 */
typedef struct {
  size_t vtcmSize;           // The VTCM for this device in Mega Byte
                             // user could not request VTCM size exceed this value
  uint32_t socModel;         // An enum value defined in Qnn Header that represent SoC model
  bool signedPdSupport;      // This field is true if the device supports Signed PD
  bool dlbcSupport;          // This field is true if the device supports DLBC
  QnnHtpDevice_Arch_t arch;  // This field shows the Architecture of this device
} QnnHtpDevice_OnChipDeviceInfoExtension_t;

/**
 * This structure is being used in QnnDevice_HardwareDeviceInfoV1_t
 * QnnDevice_getPlatformInfo use this structure to list the supported device features/info
 */
typedef struct _QnnDevice_DeviceInfoExtension_t {
  QnnHtpDevice_DeviceType_t devType;
  union UNNAMED {
    QnnHtpDevice_OnChipDeviceInfoExtension_t onChipDevice;
  };
} QnnHtpDevice_DeviceInfoExtension_t;

/**
 * @brief QNN HTP Device PerfInfrastructure specialization structure.
 *        Objects of this type are to be referenced through QnnDevice_getInfrastructure.
 *
 *        Contains function pointers for each interface method for
 *        Htp PerfInfrastructure.
 */
typedef struct {
  QnnHtpPerfInfrastructure_CreatePowerConfigIdFn_t createPowerConfigId;
  QnnHtpPerfInfrastructure_DestroyPowerConfigIdFn_t destroyPowerConfigId;
  QnnHtpPerfInfrastructure_SetPowerConfigFn_t setPowerConfig;
  QnnHtpPerfInfrastructure_SetMemoryConfigFn_t setMemoryConfig;
} QnnHtpDevice_PerfInfrastructure_t;

/// QnnHtpDevice_PerfInfrastructure_t initializer macro
#define QNN_HTP_DEVICE_PERF_INFRASTRUCTURE_INIT \
  {                                             \
    NULL,     /*createPowerConfigId*/           \
        NULL, /*destroyPowerConfigId*/          \
        NULL, /*setPowerConfig*/                \
        NULL  /*setMemoryConfig*/               \
  }

typedef enum {
  QNN_HTP_DEVICE_INFRASTRUCTURE_TYPE_PERF    = 0,
  QNN_HTP_DEVICE_INFRASTRUCTURE_TYPE_UNKNOWN = 0x7fffffff
} QnnHtpDevice_InfrastructureType_t;

typedef struct _QnnDevice_Infrastructure_t {
  QnnHtpDevice_InfrastructureType_t infraType;
  union UNNAMED {
    QnnHtpDevice_PerfInfrastructure_t perfInfra;
  };
} QnnHtpDevice_Infrastructure_t;

// clang-format on
#ifdef __cplusplus
}  // extern "C"
#endif
