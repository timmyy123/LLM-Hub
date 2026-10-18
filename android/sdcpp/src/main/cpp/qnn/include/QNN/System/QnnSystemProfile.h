//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All rights reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

/**
 *  @file
 *  @brief  QNN System Profile API.
 *
 *          This is a system API header dedicated to extensions to QnnProfile
 *          that provide backend-agnostic services to users.
 */

#ifndef QNN_SYSTEM_PROFILE_H
#define QNN_SYSTEM_PROFILE_H

#include "QnnProfile.h"
#include "QnnTypes.h"
#include "System/QnnSystemCommon.h"

#ifdef __cplusplus
extern "C" {
#endif

//=============================================================================
// Error Codes
//=============================================================================

/**
 * @brief QNN System Profile API result / error codes.
 */
typedef enum {
  /// Qnn System Profile success
  QNN_SYSTEM_PROFILE_NO_ERROR = QNN_SYSTEM_COMMON_NO_ERROR,
  /// Qnn System Profile API is not supported yet
  QNN_SYSTEM_PROFILE_ERROR_UNSUPPORTED_FEATURE = QNN_SYSTEM_COMMON_ERROR_UNSUPPORTED_FEATURE,
  /// QNN System Profile invalid handle
  QNN_SYSTEM_PROFILE_ERROR_INVALID_HANDLE = QNN_SYSTEM_COMMON_ERROR_INVALID_HANDLE,
  /// One or more arguments to a System Profile API is/are NULL/invalid.
  QNN_SYSTEM_PROFILE_ERROR_INVALID_ARGUMENT = QNN_SYSTEM_COMMON_ERROR_INVALID_ARGUMENT,

  /// QNN System Profile Specific Errors

  /// QNN System Profile writer could not allocate memory properly
  QNN_SYSTEM_PROFILE_ERROR_MEM_ALLOC = QNN_SYSTEM_PROFILE_MIN_ERROR + 0
} QnnSystemProfile_Error_t;

//=============================================================================
// Data Types
//=============================================================================

/**
 * @brief A typedef to indicate a QNN System profile handle
 */
typedef void* QnnSystemProfile_SerializationTargetHandle_t;

typedef enum {
  QNN_SYSTEM_PROFILE_SERIALIZATION_TARGET_FILE      = 0x01,
  QNN_SYSTEM_PROFILE_SERIALIZATION_TARGET_UNDEFINED = 0x7FFFFFFF
} QnnSystemProfile_SerializationTargetType_t;

typedef struct {
  const char* fileName;
  const char* fileDirectory;
} QnnSystemProfile_SerializationTargetFile_t;

typedef struct {
  QnnSystemProfile_SerializationTargetType_t type;
  union UNNAMED {
    QnnSystemProfile_SerializationTargetFile_t file;
  };
} QnnSystemProfile_SerializationTarget_t;

typedef struct {
  const char* appName;
  const char* appVersion;
  const char* backendVersion;
} QnnSystemProfile_SerializationFileHeader_t;

typedef enum {
  /// Option for maxNumMessages
  QNN_SYSTEM_PROFILE_SERIALIZATION_TARGET_CONFIG_MAX_NUM_MESSAGES = 0,
  /// Option for serializationHeader
  QNN_SYSTEM_PROFILE_SERIALIZATION_TARGET_CONFIG_SERIALIZATION_HEADER = 1,
  /// Unused, present to ensure 32 bits.
  QNN_SYSTEM_PROFILE_SERIALIZATION_TARGET_CONFIG_UNDEFINED = 0x7FFFFFFF
} QnnSystemProfile_SerializationTargetConfigType_t;

typedef struct {
  QnnSystemProfile_SerializationTargetConfigType_t type;
  union UNNAMED {
    /// The maximum number of messages before writing to a
    /// single serialzation target stops.
    uint32_t maxNumMessages;
    /// The header info that prepends all the serialized data.
    QnnSystemProfile_SerializationFileHeader_t serializationHeader;
  };
} QnnSystemProfile_SerializationTargetConfig_t;

typedef enum {
  /// Type for QnnSystemProfile_Header_t inidicating public visibility.
  QNN_SYSTEM_PROFILE_VISIBILITY_PUBLIC = 0,
  /// Type for QnnSystemProfile_Header_t inidicating private visibility.
  QNN_SYSTEM_PROFILE_VISIBILITY_PRIVATE = 1
} QnnSystemProfile_Visibility_t;

typedef enum {
  QNN_SYSTEM_PROFILE_METHOD_TYPE_NONE = 0,
  /// Backend execute method.
  QNN_SYSTEM_PROFILE_METHOD_TYPE_BACKEND_EXECUTE = 1,
  /// Backend finalize method.
  QNN_SYSTEM_PROFILE_METHOD_TYPE_BACKEND_FINALIZE = 2,
  /// Backend async execute method.
  QNN_SYSTEM_PROFILE_METHOD_TYPE_BACKEND_EXECUTE_ASYNC = 3,
  /// Backend create from binary method.
  QNN_SYSTEM_PROFILE_METHOD_TYPE_BACKEND_CREATE_FROM_BINARY = 4,
  /// Backend deinit method.
  QNN_SYSTEM_PROFILE_METHOD_TYPE_BACKEND_DEINIT = 5,
  /// App context create method.
  QNN_SYSTEM_PROFILE_METHOD_TYPE_APP_CONTEXT_CREATE = 6,
  /// App compose graphs method.
  QNN_SYSTEM_PROFILE_METHOD_TYPE_APP_COMPOSE_GRAPHS = 7,
  /// App execute inference/sec method.
  QNN_SYSTEM_PROFILE_METHOD_TYPE_APP_EXECUTE_IPS = 8,
  /// Backend graph component method.
  QNN_SYSTEM_PROFILE_METHOD_TYPE_BACKEND_GRAPH_COMPONENT = 9,
  /// App load backend library method.
  QNN_SYSTEM_PROFILE_METHOD_TYPE_APP_BACKEND_LIB_LOAD = 10,
  /// Backend apply binary method.
  QNN_SYSTEM_PROFILE_METHOD_TYPE_BACKEND_APPLY_BINARY_SECTION = 11,
  /// Context finalize method.
  QNN_SYSTEM_PROFILE_METHOD_TYPE_CONTEXT_FINALIZE = 12,
  /// Context get binary size method.
  QNN_SYSTEM_PROFILE_METHOD_TYPE_CONTEXT_GET_BINARY_SIZE = 13,
  /// Context get binary method.
  QNN_SYSTEM_PROFILE_METHOD_TYPE_CONTEXT_GET_BINARY = 14,
  /// Context get binary section size method.
  QNN_SYSTEM_PROFILE_METHOD_TYPE_CONTEXT_GET_BINARY_SECTION_SIZE = 15,
  // Backend finalize method performed after tensor updates.
  QNN_SYSTEM_PROFILE_METHOD_TYPE_BACKEND_FINALIZE_TENSOR_UPDATES = 16,
  /// Backend update binary section method.
  QNN_SYSTEM_PROFILE_METHOD_TYPE_BACKEND_UPDATE_BINARY_SECTION = 17
} QnnSystemProfile_MethodType_t;

typedef struct {
  uint64_t startTime;
  uint64_t stopTime;
  uint64_t startMem;
  uint64_t stopMem;
  QnnSystemProfile_MethodType_t methodType;
  QnnSystemProfile_Visibility_t visibility;
  const char* graphName;
} QnnSystemProfile_HeaderV1_t;

// clang-format off
/// QnnSystemProfile_HeaderV1_t initializer macro
#define QNN_SYSTEM_PROFILE_HEADER_V1_INIT                   \
  {                                                         \
    0,                                    /* startTime */   \
    0,                                    /* stopTime */    \
    0,                                    /* startMem */    \
    0,                                    /* stopMem */     \
    QNN_SYSTEM_PROFILE_METHOD_TYPE_NONE,  /* methodType */  \
    QNN_SYSTEM_PROFILE_VISIBILITY_PUBLIC, /* visibility */  \
    NULL,                                 /* graphName */     \
  }
// clang-format on

typedef enum {
  /// Type for QnnSystemProfile_ProfileDataV1_t containing eventData
  QNN_SYSTEM_PROFILE_EVENT_DATA = 0,
  /// Type for QnnSystemProfile_ProfileDataV1_t containing extendedEventData
  QNN_SYSTEM_PROFILE_EXTENDED_EVENT_DATA = 1,
  /// Unused, present to ensure 32 bits.
  QNN_SYSTEM_PROFILE_EVENT_DATA_UNDEFINED = 0x7FFFFFFF
} QnnSystemProfile_EventDataType_t;

typedef struct QnnSystemProfile_ProfileEventV1_t QnnSystemProfile_ProfileEventV1_t;

struct QnnSystemProfile_ProfileEventV1_t {
  QnnSystemProfile_EventDataType_t type;
  union {
    QnnProfile_EventData_t eventData;
    QnnProfile_ExtendedEventData_t extendedEventData;
  };
  QnnSystemProfile_ProfileEventV1_t* profileSubEventData;
  uint32_t numSubEvents;
};

// clang-format off
/// QnnSystemProfile_ProfileEventV1_t initializer macro
#define QNN_SYSTEM_PROFILE_EVENT_V1_INIT                               \
  {                                                                    \
    QNN_SYSTEM_PROFILE_EVENT_DATA_UNDEFINED, /* type */                \
    {                                                                  \
      QNN_PROFILE_EVENT_DATA_INIT            /* eventData */           \
    },                                                                 \
    NULL,                                    /* profileSubEventData */ \
    0                                        /* numSubEvents */        \
  }
// clang-format on

typedef struct {
  QnnSystemProfile_HeaderV1_t header;
  QnnSystemProfile_ProfileEventV1_t* profilingEvents;
  uint32_t numProfilingEvents;
} QnnSystemProfile_ProfileDataV1_t;

// clang-format off
/// QnnSystemProfile_ProfileDataV1_t initializer macro
#define QNN_SYSTEM_PROFILE_DATA_V1_INIT                         \
  {                                                             \
    QNN_SYSTEM_PROFILE_HEADER_V1_INIT, /* header */             \
    NULL,                              /* profilingEvents */    \
    0                                  /* numProfilingEvents */ \
  }
// clang-format on

typedef enum {
  /// Version type to access v1
  QNN_SYSTEM_PROFILE_DATA_VERSION_1 = 0x01,
  /// Unused, present to ensure 32 bits.
  QNN_SYSTEM_PROFILE_DATA_VERSION_UNDEFINED = 0x7FFFFFFF
} QnnSystemProfile_ProfileDataVersion_t;

typedef struct {
  QnnSystemProfile_ProfileDataVersion_t version;
  union UNNAMED {
    QnnSystemProfile_ProfileDataV1_t v1;
  };
} QnnSystemProfile_ProfileData_t;

// clang-format off
/// QnnSystemProfile_ProfileData_t initializer macro
#define QNN_SYSTEM_PROFILE_DATA_INIT                         \
  {                                                          \
    QNN_SYSTEM_PROFILE_DATA_VERSION_UNDEFINED, /* version */ \
    {                                                        \
      QNN_SYSTEM_PROFILE_DATA_V1_INIT          /* v1 */      \
    },                                                       \
  }
// clang-format on

/**
 * Function to create a serialization target __serializationTarget__ from the information in
 * __serializationTargetInfo__.
 * @param[in] serializationTargetConfig the information about the serialization target
 * @param[in] configs the configurations for the serialization target
 * @param[in] numConfigs the number of configuration options
 * @param[out] serializationTarget handle to the serialization target. For use in subsequent calls .
 */
QNN_SYSTEM_API
Qnn_ErrorHandle_t QnnSystemProfile_createSerializationTarget(
    QnnSystemProfile_SerializationTarget_t serializationTargetInfo,
    QnnSystemProfile_SerializationTargetConfig_t* configs,
    uint32_t numConfigs,
    QnnSystemProfile_SerializationTargetHandle_t* serializationTarget);

/**
 * Function to serialize and write provided event data __eventData__ to a serialization target
 * __serializationTarget__.
 * @param[in] serializationTarget  the target to write the output to.
 * @param[in] eventData the array of pointers to event data to serialize
 * @param[in] numEvents the number of events to serialize
 */
QNN_SYSTEM_API
Qnn_ErrorHandle_t QnnSystemProfile_serializeEventData(
    QnnSystemProfile_SerializationTargetHandle_t serializationTarget,
    const QnnSystemProfile_ProfileData_t** eventData,
    uint32_t numEvents);

/**
 * Function to free a system serialization target.
 * @param[in] serializationTarget handle to free the serialization target.
 */
QNN_SYSTEM_API
Qnn_ErrorHandle_t QnnSystemProfile_freeSerializationTarget(
    QnnSystemProfile_SerializationTargetHandle_t serializationTarget);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // QNN_SYSTEM_PROFILE_H