//==============================================================================
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// All rights reserved.
// Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

/**
 *  @file
 *  @brief  QNN System Interface API
 *
 *          QNN System Interface is an abstraction combining all QNN System APIs.
 *          QNN System Interface provides typedef variant of QNN System APIs and
 *          API to get QNN System interface object(s).
 *          QNN System Interface API can coexist with QNN System APIs. Visibility
 *          of Interface and System APIs is determined by build configuration,
 *          specifically by QNN_SYSTEM_API and QNN_SYSTEM_INTERFACE macro definitions.
 */

#ifndef QNN_SYSTEM_INTERFACE_H
#define QNN_SYSTEM_INTERFACE_H

#include "System/QnnSystemCommon.h"

// QNN System API headers
#include "System/QnnSystemContext.h"
#include "System/QnnSystemDlc.h"
#include "System/QnnSystemLog.h"
#include "System/QnnSystemProfile.h"
#include "System/QnnSystemTensor.h"

#ifdef __cplusplus
extern "C" {
#endif

//=============================================================================
// Macros
//=============================================================================

// Macro controlling visibility of QNN System Interface API
#ifndef QNN_SYSTEM_INTERFACE
#define QNN_SYSTEM_INTERFACE
#endif

// Utility macros for version and name construction
#define QNN_SYSTEM_INTERFACE_VER_EVAL(major, minor)          QNN_PASTE_THREE(major, _, minor)
#define QNN_SYSTEM_INTERFACE_NAME_EVAL(prefix, body, suffix) QNN_PASTE_THREE(prefix, body, suffix)

// Construct interface type name from version, e.g. QnnSystemInterface_ImplementationV0_0_t
#define QNN_SYSTEM_INTERFACE_VER_TYPE_EVAL(ver_major, ver_minor) \
  QNN_SYSTEM_INTERFACE_NAME_EVAL(                                \
      QnnSystemInterface_ImplementationV, QNN_SYSTEM_INTERFACE_VER_EVAL(ver_major, ver_minor), _t)

// Construct interface name from version, e.g. v0_0
#define QNN_SYSTEM_INTERFACE_VER_NAME_EVAL(ver_major, ver_minor) \
  QNN_SYSTEM_INTERFACE_NAME_EVAL(v, QNN_SYSTEM_INTERFACE_VER_EVAL(ver_major, ver_minor), )

// Interface type name for current API version
#define QNN_SYSTEM_INTERFACE_VER_TYPE \
  QNN_SYSTEM_INTERFACE_VER_TYPE_EVAL(QNN_SYSTEM_API_VERSION_MAJOR, QNN_SYSTEM_API_VERSION_MINOR)

// Interface name for current API version
#define QNN_SYSTEM_INTERFACE_VER_NAME \
  QNN_SYSTEM_INTERFACE_VER_NAME_EVAL(QNN_SYSTEM_API_VERSION_MAJOR, QNN_SYSTEM_API_VERSION_MINOR)

//=============================================================================
// Data Types
//=============================================================================

/**
 * @brief QNN System Interface API result / error codes
 */
typedef enum {
  QNN_SYSTEM_INTERFACE_MIN_ERROR = QNN_MIN_ERROR_SYSTEM,
  ////////////////////////////////////////

  QNN_SYSTEM_INTERFACE_NO_ERROR                = QNN_SUCCESS,
  QNN_SYSTEM_INTERFACE_ERROR_NOT_SUPPORTED     = QNN_COMMON_ERROR_NOT_SUPPORTED,
  QNN_SYSTEM_INTERFACE_ERROR_INVALID_PARAMETER = QNN_COMMON_ERROR_INVALID_ARGUMENT,

  ////////////////////////////////////////
  QNN_SYSTEM_INTERFACE_MAX_ERROR = QNN_MAX_ERROR_SYSTEM
} QnnSystemInterface_Error_t;

//
// From QnnSystemContext.h
//

/** @brief See QnnSystemContext_create()*/
typedef Qnn_ErrorHandle_t (*QnnSystemContext_CreateFn_t)(QnnSystemContext_Handle_t* sysCtxHandle);

/** @brief See QnnSystemContext_getBinaryInfo()*/
typedef Qnn_ErrorHandle_t (*QnnSystemContext_GetBinaryInfoFn_t)(
    QnnSystemContext_Handle_t sysCtxHandle,
    void* binaryBuffer,
    uint64_t binaryBufferSize,
    const QnnSystemContext_BinaryInfo_t** binaryInfo,
    Qnn_ContextBinarySize_t* binaryInfoSize);

/** @brief See QnnSystemContext_getMetadata()*/
typedef Qnn_ErrorHandle_t (*QnnSystemContext_GetMetaDataFn_t)(
    QnnSystemContext_Handle_t sysCtxHandle,
    const void* binaryBuffer,
    uint64_t binaryBufferSize,
    const QnnSystemContext_BinaryInfo_t** binaryInfo);

/** @brief See QnnSystemContext_free()*/
typedef Qnn_ErrorHandle_t (*QnnSystemContext_FreeFn_t)(QnnSystemContext_Handle_t sysCtxHandle);

//
// From QnnSystemTensor.h
//

/** @brief See QnnSystemTensor_getMemoryFootprint()*/
typedef Qnn_ErrorHandle_t (*QnnSystemTensor_getMemoryFootprintFn_t)(Qnn_Tensor_t tensor,
                                                                    uint64_t* footprint);

//
// From QnnSystemLog.h
//

/** @brief See QnnSystemLog_create()*/
typedef Qnn_ErrorHandle_t (*QnnSystemLog_createFn_t)(QnnLog_Callback_t callback,
                                                     QnnLog_Level_t maxLogLevel,
                                                     Qnn_LogHandle_t* logger);

/** @brief See QnnSystemLog_setLogLevel()*/
typedef Qnn_ErrorHandle_t (*QnnSystemLog_setLogLevelFn_t)(Qnn_LogHandle_t logger,
                                                          QnnLog_Level_t maxLogLevel);

/** @brief See QnnSystemLog_free()*/
typedef Qnn_ErrorHandle_t (*QnnSystemLog_freeFn_t)(Qnn_LogHandle_t logger);
// clang-format off

//
// From QnnSystemDlc.h
//

/** @brief See QnnSystemDlc_createWithDestinationDir()*/
typedef Qnn_ErrorHandle_t (*QnnSystemDlc_createWithDestinationDirFn_t)(
    Qnn_LogHandle_t logger,
    const char* destinationDirectoryHint,
    QnnSystemDlc_Handle_t* dlcHandle);

/** @brief See QnnSystemDlc_createFromFile()*/
typedef Qnn_ErrorHandle_t (*QnnSystemDlc_createFromFileFn_t)(Qnn_LogHandle_t logger,
                                                             const char* dlcPath,
                                                             QnnSystemDlc_Handle_t* dlcHandle);
/** @brief See QnnSystemDlc_createFromBinary()*/
typedef Qnn_ErrorHandle_t (*QnnSystemDlc_createFromBinaryFn_t)(Qnn_LogHandle_t logger,
                                                               const uint8_t* buffer,
                                                               const Qnn_ContextBinarySize_t bufferSize,
                                                               QnnSystemDlc_Handle_t* dlcHandle);

/** @brief See QnnSystemDlc_composeGraphs()*/
typedef Qnn_ErrorHandle_t (*QnnSystemDlc_composeGraphsFn_t)(QnnSystemDlc_Handle_t dlcHandle,
                                                            const QnnSystemDlc_GraphConfigInfo_t** graphConfigs,
                                                            const uint32_t numGraphConfigs,
                                                            Qnn_BackendHandle_t backend,
                                                            Qnn_ContextHandle_t context,
                                                            QnnInterface_t backendInterface,
                                                            QnnSystemContext_GraphInfoVersion_t graphVersion,
                                                            QnnSystemContext_GraphInfo_t** graphs,
                                                            uint32_t* numGraphs);
/** @brief See QnnSystemDlc_getOpMappings()*/
typedef Qnn_ErrorHandle_t (*QnnSystemDlc_getOpMappingsFn_t)(QnnSystemDlc_Handle_t dlcHandle,
                                                          const Qnn_OpMapping_t** opMappings,
                                                          uint32_t* numOpMappings);

/** @brief See QnnSystemDlc_getRecordByName()*/
typedef Qnn_ErrorHandle_t (*QnnSystemDlc_getRecordByNameFn_t)(QnnSystemDlc_Handle_t dlcHandle,
                                                              const char* recordName,
                                                              QnnSystemDlc_RecordHandle_t* recordHandle);

/** @brief See QnnSystemDlc_getRecordsByType()*/
typedef Qnn_ErrorHandle_t (*QnnSystemDlc_getRecordsByTypeFn_t)(QnnSystemDlc_Handle_t dlcHandle,
                                                               QnnSystemDlc_RecordType_t recordType,
                                                               uint8_t getMostOptimalContextBinary,
                                                               QnnSystemDlc_RecordHandle_t** recordHandles,
                                                               uint32_t* numRecordHandles);

/** @brief See QnnSystemDlc_getRecordDataSize()*/
typedef Qnn_ErrorHandle_t (*QnnSystemDlc_getRecordDataSizeFn_t)(QnnSystemDlc_RecordHandle_t recordHandle,
                                                                uint64_t* dataSize);

/** @brief See QnnSystemDlc_readRecordDataMemoryMapped()*/
typedef Qnn_ErrorHandle_t (*QnnSystemDlc_readRecordDataMemoryMappedFn_t)(QnnSystemDlc_RecordHandle_t recordHandle,
                                                                         const uint8_t** data,
                                                                         uint64_t* dataSize);

/** @brief See QnnSystemDlc_freeRecord()*/
typedef Qnn_ErrorHandle_t (*QnnSystemDlc_freeRecordFn_t)(QnnSystemDlc_RecordHandle_t recordHandle);

/** @brief See QnnSystemDlc_free()*/
typedef Qnn_ErrorHandle_t (*QnnSystemDlc_freeFn_t)(QnnSystemDlc_Handle_t dlcHandle);

/** @brief See QnnSystemDlc_save()*/
typedef Qnn_ErrorHandle_t (*QnnSystemDlc_saveFn_t)(QnnSystemDlc_Handle_t dlcHandle,
                                                   const char* path);

/** @brief See QnnSystemDlc_createFromFileWithDestinationDir()*/
typedef Qnn_ErrorHandle_t (*QnnSystemDlc_createFromFileWithDestinationDirFn_t)(
    Qnn_LogHandle_t logger,
    const char* dlcPath,
    const char* destinationDirectoryHint,
    QnnSystemDlc_Handle_t* dlcHandle);

/** @brief See QnnSystemDlc_createFromBinaryWithDestinationDir()*/
typedef Qnn_ErrorHandle_t (*QnnSystemDlc_createFromBinaryWithDestinationDirFn_t)(
    Qnn_LogHandle_t logger,
    const uint8_t* buffer,
    const Qnn_ContextBinarySize_t bufferSize,
    const char* destinationDirectoryHint,
    QnnSystemDlc_Handle_t* dlcHandle);

/** @brief See QnnSystemDlc_getBinarySize()*/
typedef Qnn_ErrorHandle_t (*QnnSystemDlc_getBinarySizeFn_t)(
    QnnSystemDlc_Handle_t dlcHandle,
    Qnn_SystemDlcBinarySize_t* binaryBufferSize);

/** @brief See QnnSystemDlc_getBinary()*/
typedef Qnn_ErrorHandle_t (*QnnSystemDlc_getBinaryFn_t)(
    QnnSystemDlc_Handle_t dlcHandle,
    uint8_t* binaryBuffer,
    Qnn_SystemDlcBinarySize_t binaryBufferSize,
    Qnn_SystemDlcBinarySize_t* writtenBufferSize);

/** @brief See QnnSystemDlc_getComposableGraphInfoHandles()*/
typedef Qnn_ErrorHandle_t (*QnnSystemDlc_getComposableGraphInfoHandlesFn_t)(
    QnnSystemDlc_Handle_t dlcHandle,
    QnnSystemDlc_GraphInfoHandle_t** graphInfoHandles,
    uint32_t* numGraphInfoHandles);

/** @brief See QnnSystemDlc_getContextBinaryInfoHandles()*/
typedef Qnn_ErrorHandle_t (*QnnSystemDlc_getContextBinaryInfoHandlesFn_t)(
    QnnSystemDlc_Handle_t dlcHandle,
    QnnSystemDlc_ContextBinaryInfoHandle_t** contextBinaryInfoHandles,
    uint32_t* numContextBinaryInfoHandles);

/** @brief See QnnSystemDlc_getComposableGraphInfo()*/
typedef Qnn_ErrorHandle_t (*QnnSystemDlc_getComposableGraphInfoFn_t)(
    QnnSystemDlc_GraphInfoHandle_t graphInfoHandle,
    const QnnSystemDlc_ComposableGraphInfo_t** graphInfo);

/** @brief See QnnSystemDlc_getContextBinaryInfo()*/
typedef Qnn_ErrorHandle_t (*QnnSystemDlc_getContextBinaryInfoFn_t)(
    QnnSystemDlc_ContextBinaryInfoHandle_t contextBinaryInfoHandle,
    const QnnSystemContext_BinaryInfo_t** binaryInfo);

/** @brief See QnnSystemDlc_getComposableGraphInfoGraphName()*/
typedef Qnn_ErrorHandle_t (*QnnSystemDlc_getComposableGraphInfoGraphNameFn_t)(
    QnnSystemDlc_GraphInfoHandle_t graphInfoHandle, const char** graphName);

/** @brief See QnnSystemDlc_getComposableGraphInfoNumInputTensors()*/
typedef Qnn_ErrorHandle_t (*QnnSystemDlc_getComposableGraphInfoNumInputTensorsFn_t)(
    QnnSystemDlc_GraphInfoHandle_t graphInfoHandle, uint32_t* numInputTensors);

/** @brief See QnnSystemDlc_getComposableGraphInfoNumOutputTensors()*/
typedef Qnn_ErrorHandle_t (*QnnSystemDlc_getComposableGraphInfoNumOutputTensorsFn_t)(
    QnnSystemDlc_GraphInfoHandle_t graphInfoHandle, uint32_t* numOutputTensors);

/** @brief See QnnSystemDlc_getGraphInfoInputTensorInfo()*/
typedef Qnn_ErrorHandle_t (*QnnSystemDlc_getGraphInfoInputTensorInfoFn_t)(
    QnnSystemDlc_GraphInfoHandle_t graphInfoHandle,
    uint32_t index,
    QnnSystemDlc_TensorInfoHandle_t* tensorHandle);

/** @brief See QnnSystemDlc_getGraphInfoOutputTensorInfo()*/
typedef Qnn_ErrorHandle_t (*QnnSystemDlc_getGraphInfoOutputTensorInfoFn_t)(
    QnnSystemDlc_GraphInfoHandle_t graphInfoHandle,
    uint32_t index,
    QnnSystemDlc_TensorInfoHandle_t* tensorHandle);

/** @brief See QnnSystemDlc_getTensorInfoName()*/
typedef Qnn_ErrorHandle_t (*QnnSystemDlc_getTensorInfoNameFn_t)(
    QnnSystemDlc_TensorInfoHandle_t tensorHandle, const char** name);

/** @brief See QnnSystemDlc_getTensorInfoId()*/
typedef Qnn_ErrorHandle_t (*QnnSystemDlc_getTensorInfoIdFn_t)(
    QnnSystemDlc_TensorInfoHandle_t tensorHandle, uint32_t* id);

/** @brief See QnnSystemDlc_getTensorInfoType()*/
typedef Qnn_ErrorHandle_t (*QnnSystemDlc_getTensorInfoTypeFn_t)(
    QnnSystemDlc_TensorInfoHandle_t tensorHandle, Qnn_TensorType_t* type);

/** @brief See QnnSystemDlc_getTensorInfoDataFormat()*/
typedef Qnn_ErrorHandle_t (*QnnSystemDlc_getTensorInfoDataFormatFn_t)(
    QnnSystemDlc_TensorInfoHandle_t tensorHandle, Qnn_TensorDataFormat_t* dataFormat);

/** @brief See QnnSystemDlc_getTensorInfoDataType()*/
typedef Qnn_ErrorHandle_t (*QnnSystemDlc_getTensorInfoDataTypeFn_t)(
    QnnSystemDlc_TensorInfoHandle_t tensorHandle, Qnn_DataType_t* dataType);

/** @brief See QnnSystemDlc_getTensorInfoRank()*/
typedef Qnn_ErrorHandle_t (*QnnSystemDlc_getTensorInfoRankFn_t)(
    QnnSystemDlc_TensorInfoHandle_t tensorHandle, uint32_t* rank);

/** @brief See QnnSystemDlc_getTensorInfoDimensions()*/
typedef Qnn_ErrorHandle_t (*QnnSystemDlc_getTensorInfoDimensionsFn_t)(
    QnnSystemDlc_TensorInfoHandle_t tensorHandle, uint32_t* dimensions);

/** @brief See QnnSystemDlc_getTensorInfoQuantParams()*/
typedef Qnn_ErrorHandle_t (*QnnSystemDlc_getTensorInfoQuantParamsFn_t)(
    QnnSystemDlc_TensorInfoHandle_t tensorHandle,
    QnnSystemDlc_QuantParamsHandle_t* quantParamsHandle);

/** @brief See QnnSystemDlc_getQuantParamsEncodingDefinition()*/
typedef Qnn_ErrorHandle_t (*QnnSystemDlc_getQuantParamsEncodingDefinitionFn_t)(
    QnnSystemDlc_QuantParamsHandle_t quantParamsHandle, Qnn_Definition_t* encodingDefinition);

/** @brief See QnnSystemDlc_getQuantParamsEncoding()*/
typedef Qnn_ErrorHandle_t (*QnnSystemDlc_getQuantParamsEncodingFn_t)(
    QnnSystemDlc_QuantParamsHandle_t quantParamsHandle, Qnn_QuantizationEncoding_t* encoding);

/** @brief See QnnSystemDlc_getQuantParamsScaleOffsetScale()*/
typedef Qnn_ErrorHandle_t (*QnnSystemDlc_getQuantParamsScaleOffsetScaleFn_t)(
    QnnSystemDlc_QuantParamsHandle_t quantParamsHandle, float* scale);

/** @brief See QnnSystemDlc_getQuantParamsScaleOffsetOffset()*/
typedef Qnn_ErrorHandle_t (*QnnSystemDlc_getQuantParamsScaleOffsetOffsetFn_t)(
    QnnSystemDlc_QuantParamsHandle_t quantParamsHandle, int32_t* offset);

/** @brief See QnnSystemDlc_getQuantParamsBwScaleOffsetBitwidth()*/
typedef Qnn_ErrorHandle_t (*QnnSystemDlc_getQuantParamsBwScaleOffsetBitwidthFn_t)(
    QnnSystemDlc_QuantParamsHandle_t quantParamsHandle, uint32_t* bitwidth);

/** @brief See QnnSystemDlc_getQuantParamsBwScaleOffsetScale()*/
typedef Qnn_ErrorHandle_t (*QnnSystemDlc_getQuantParamsBwScaleOffsetScaleFn_t)(
    QnnSystemDlc_QuantParamsHandle_t quantParamsHandle, float* scale);

/** @brief See QnnSystemDlc_getQuantParamsBwScaleOffsetOffset()*/
typedef Qnn_ErrorHandle_t (*QnnSystemDlc_getQuantParamsBwScaleOffsetOffsetFn_t)(
    QnnSystemDlc_QuantParamsHandle_t quantParamsHandle, int32_t* offset);

/** @brief See QnnSystemDlc_getQuantParamsAxisScaleOffsetAxis()*/
typedef Qnn_ErrorHandle_t (*QnnSystemDlc_getQuantParamsAxisScaleOffsetAxisFn_t)(
    QnnSystemDlc_QuantParamsHandle_t quantParamsHandle, int32_t* axis);

/** @brief See QnnSystemDlc_getQuantParamsAxisScaleOffsetNumScaleOffsets()*/
typedef Qnn_ErrorHandle_t (*QnnSystemDlc_getQuantParamsAxisScaleOffsetNumScaleOffsetsFn_t)(
    QnnSystemDlc_QuantParamsHandle_t quantParamsHandle, uint32_t* numScaleOffsets);

/** @brief See QnnSystemDlc_getQuantParamsAxisScaleOffsetScaleAtIndex()*/
typedef Qnn_ErrorHandle_t (*QnnSystemDlc_getQuantParamsAxisScaleOffsetScaleAtIndexFn_t)(
    QnnSystemDlc_QuantParamsHandle_t quantParamsHandle, uint32_t index, float* scale);

/** @brief See QnnSystemDlc_getQuantParamsAxisScaleOffsetOffsetAtIndex()*/
typedef Qnn_ErrorHandle_t (*QnnSystemDlc_getQuantParamsAxisScaleOffsetOffsetAtIndexFn_t)(
    QnnSystemDlc_QuantParamsHandle_t quantParamsHandle, uint32_t index, int32_t* offset);

/** @brief See QnnSystemDlc_getQuantParamsBwAxisScaleOffsetBitwidth()*/
typedef Qnn_ErrorHandle_t (*QnnSystemDlc_getQuantParamsBwAxisScaleOffsetBitwidthFn_t)(
    QnnSystemDlc_QuantParamsHandle_t quantParamsHandle, uint32_t* bitwidth);

/** @brief See QnnSystemDlc_getQuantParamsBwAxisScaleOffsetAxis()*/
typedef Qnn_ErrorHandle_t (*QnnSystemDlc_getQuantParamsBwAxisScaleOffsetAxisFn_t)(
    QnnSystemDlc_QuantParamsHandle_t quantParamsHandle, int32_t* axis);

/** @brief See QnnSystemDlc_getQuantParamsBwAxisScaleOffsetNumElements()*/
typedef Qnn_ErrorHandle_t (*QnnSystemDlc_getQuantParamsBwAxisScaleOffsetNumElementsFn_t)(
    QnnSystemDlc_QuantParamsHandle_t quantParamsHandle, uint32_t* numElements);

/** @brief See QnnSystemDlc_getQuantParamsBwAxisScaleOffsetScaleAtIndex()*/
typedef Qnn_ErrorHandle_t (*QnnSystemDlc_getQuantParamsBwAxisScaleOffsetScaleAtIndexFn_t)(
    QnnSystemDlc_QuantParamsHandle_t quantParamsHandle, uint32_t index, float* scale);

/** @brief See QnnSystemDlc_getQuantParamsBwAxisScaleOffsetOffsetAtIndex()*/
typedef Qnn_ErrorHandle_t (*QnnSystemDlc_getQuantParamsBwAxisScaleOffsetOffsetAtIndexFn_t)(
    QnnSystemDlc_QuantParamsHandle_t quantParamsHandle, uint32_t index, int32_t* offset);

/** @brief See QnnSystemDlc_getContextBinaryInfoBackendId()*/
typedef Qnn_ErrorHandle_t (*QnnSystemDlc_getContextBinaryInfoBackendIdFn_t)(
    QnnSystemDlc_ContextBinaryInfoHandle_t handle, uint32_t* backendId);
/** @brief See QnnSystemDlc_getContextBinaryInfoBuildId()*/
typedef Qnn_ErrorHandle_t (*QnnSystemDlc_getContextBinaryInfoBuildIdFn_t)(
    QnnSystemDlc_ContextBinaryInfoHandle_t handle, const char** buildId);
/** @brief See QnnSystemDlc_getContextBinaryInfoCoreApiVersion()*/
typedef Qnn_ErrorHandle_t (*QnnSystemDlc_getContextBinaryInfoCoreApiVersionFn_t)(
    QnnSystemDlc_ContextBinaryInfoHandle_t handle, uint32_t* major, uint32_t* minor, uint32_t* patch);
/** @brief See QnnSystemDlc_getContextBinaryInfoBackendApiVersion()*/
typedef Qnn_ErrorHandle_t (*QnnSystemDlc_getContextBinaryInfoBackendApiVersionFn_t)(
    QnnSystemDlc_ContextBinaryInfoHandle_t handle, uint32_t* major, uint32_t* minor, uint32_t* patch);
/** @brief See QnnSystemDlc_getContextBinaryInfoSocVersion()*/
typedef Qnn_ErrorHandle_t (*QnnSystemDlc_getContextBinaryInfoSocVersionFn_t)(
    QnnSystemDlc_ContextBinaryInfoHandle_t handle, const char** socVersion);
/** @brief See QnnSystemDlc_getContextBinaryInfoHwInfoBlobVersion()*/
typedef Qnn_ErrorHandle_t (*QnnSystemDlc_getContextBinaryInfoHwInfoBlobVersionFn_t)(
    QnnSystemDlc_ContextBinaryInfoHandle_t handle, uint32_t* major, uint32_t* minor, uint32_t* patch);
/** @brief See QnnSystemDlc_getContextBinaryInfoContextBlobVersion()*/
typedef Qnn_ErrorHandle_t (*QnnSystemDlc_getContextBinaryInfoContextBlobVersionFn_t)(
    QnnSystemDlc_ContextBinaryInfoHandle_t handle, uint32_t* major, uint32_t* minor, uint32_t* patch);
/** @brief See QnnSystemDlc_getContextBinaryInfoHwInfoBlobSize()*/
typedef Qnn_ErrorHandle_t (*QnnSystemDlc_getContextBinaryInfoHwInfoBlobSizeFn_t)(
    QnnSystemDlc_ContextBinaryInfoHandle_t handle, uint32_t* hwInfoBlobSize);
/** @brief See QnnSystemDlc_getContextBinaryInfoHwInfoBlob()*/
typedef Qnn_ErrorHandle_t (*QnnSystemDlc_getContextBinaryInfoHwInfoBlobFn_t)(
    QnnSystemDlc_ContextBinaryInfoHandle_t handle, void** hwInfoBlob);
/** @brief See QnnSystemDlc_getContextBinaryInfoContextBlobSize()*/
typedef Qnn_ErrorHandle_t (*QnnSystemDlc_getContextBinaryInfoContextBlobSizeFn_t)(
    QnnSystemDlc_ContextBinaryInfoHandle_t handle, uint64_t* contextBlobSize);
/** @brief See QnnSystemDlc_getContextBinaryInfoNumContextTensors()*/
typedef Qnn_ErrorHandle_t (*QnnSystemDlc_getContextBinaryInfoNumContextTensorsFn_t)(
    QnnSystemDlc_ContextBinaryInfoHandle_t handle, uint32_t* numContextTensors);
/** @brief See QnnSystemDlc_getContextBinaryInfoContextTensorInfo()*/
typedef Qnn_ErrorHandle_t (*QnnSystemDlc_getContextBinaryInfoContextTensorInfoFn_t)(
    QnnSystemDlc_ContextBinaryInfoHandle_t handle, uint32_t index,
    QnnSystemDlc_TensorInfoHandle_t* tensorInfoHandle);
/** @brief See QnnSystemDlc_getContextBinaryInfoNumGraphs()*/
typedef Qnn_ErrorHandle_t (*QnnSystemDlc_getContextBinaryInfoNumGraphsFn_t)(
    QnnSystemDlc_ContextBinaryInfoHandle_t handle, uint32_t* numGraphs);
/** @brief See QnnSystemDlc_getContextGraphInfoHandle()*/
typedef Qnn_ErrorHandle_t (*QnnSystemDlc_getContextGraphInfoHandleFn_t)(
    QnnSystemDlc_ContextBinaryInfoHandle_t handle, uint32_t index,
    QnnSystemDlc_ContextGraphInfoHandle_t* contextGraphInfoHandle);
/** @brief See QnnSystemDlc_getContextBinaryInfoSocModel()*/
typedef Qnn_ErrorHandle_t (*QnnSystemDlc_getContextBinaryInfoSocModelFn_t)(
    QnnSystemDlc_ContextBinaryInfoHandle_t handle, uint32_t* socModel);
/** @brief See QnnSystemDlc_getContextBinaryInfoContextMetadataSize()*/
typedef Qnn_ErrorHandle_t (*QnnSystemDlc_getContextBinaryInfoContextMetadataSizeFn_t)(
    QnnSystemDlc_ContextBinaryInfoHandle_t handle, uint32_t* contextMetadataSize);
/** @brief See QnnSystemDlc_getContextBinaryInfoContextMetadata()*/
typedef Qnn_ErrorHandle_t (*QnnSystemDlc_getContextBinaryInfoContextMetadataFn_t)(
    QnnSystemDlc_ContextBinaryInfoHandle_t handle, void** contextMetadata);
/** @brief See QnnSystemDlc_getContextGraphInfoGraphName()*/
typedef Qnn_ErrorHandle_t (*QnnSystemDlc_getContextGraphInfoGraphNameFn_t)(
    QnnSystemDlc_ContextGraphInfoHandle_t handle, const char** graphName);
/** @brief See QnnSystemDlc_getContextGraphInfoNumInputTensors()*/
typedef Qnn_ErrorHandle_t (*QnnSystemDlc_getContextGraphInfoNumInputTensorsFn_t)(
    QnnSystemDlc_ContextGraphInfoHandle_t handle, uint32_t* numInputTensors);
/** @brief See QnnSystemDlc_getContextGraphInfoNumOutputTensors()*/
typedef Qnn_ErrorHandle_t (*QnnSystemDlc_getContextGraphInfoNumOutputTensorsFn_t)(
    QnnSystemDlc_ContextGraphInfoHandle_t handle, uint32_t* numOutputTensors);
/** @brief See QnnSystemDlc_getContextGraphInfoNumUpdateableTensors()*/
typedef Qnn_ErrorHandle_t (*QnnSystemDlc_getContextGraphInfoNumUpdateableTensorsFn_t)(
    QnnSystemDlc_ContextGraphInfoHandle_t handle, uint32_t* numUpdateableTensors);
/** @brief See QnnSystemDlc_getContextGraphInfoGraphBlobInfoSize()*/
typedef Qnn_ErrorHandle_t (*QnnSystemDlc_getContextGraphInfoGraphBlobInfoSizeFn_t)(
    QnnSystemDlc_ContextGraphInfoHandle_t handle, uint32_t* graphBlobInfoSize);
/** @brief See QnnSystemDlc_getContextGraphInfoGraphBlobInfo()*/
typedef Qnn_ErrorHandle_t (*QnnSystemDlc_getContextGraphInfoGraphBlobInfoFn_t)(
    QnnSystemDlc_ContextGraphInfoHandle_t handle, void** graphBlobInfo);
/** @brief See QnnSystemDlc_getContextGraphInfoStartOpIndex()*/
typedef Qnn_ErrorHandle_t (*QnnSystemDlc_getContextGraphInfoStartOpIndexFn_t)(
    QnnSystemDlc_ContextGraphInfoHandle_t handle, uint32_t* startOpIndex);
/** @brief See QnnSystemDlc_getContextGraphInfoEndOpIndex()*/
typedef Qnn_ErrorHandle_t (*QnnSystemDlc_getContextGraphInfoEndOpIndexFn_t)(
    QnnSystemDlc_ContextGraphInfoHandle_t handle, uint32_t* endOpIndex);
/** @brief See QnnSystemDlc_getContextGraphInfoInputTensorInfo()*/
typedef Qnn_ErrorHandle_t (*QnnSystemDlc_getContextGraphInfoInputTensorInfoFn_t)(
    QnnSystemDlc_ContextGraphInfoHandle_t handle, uint32_t index,
    QnnSystemDlc_TensorInfoHandle_t* tensorInfoHandle);
/** @brief See QnnSystemDlc_getContextGraphInfoOutputTensorInfo()*/
typedef Qnn_ErrorHandle_t (*QnnSystemDlc_getContextGraphInfoOutputTensorInfoFn_t)(
    QnnSystemDlc_ContextGraphInfoHandle_t handle, uint32_t index,
    QnnSystemDlc_TensorInfoHandle_t* tensorInfoHandle);
/** @brief See QnnSystemDlc_getContextGraphInfoUpdateableTensorInfo()*/
typedef Qnn_ErrorHandle_t (*QnnSystemDlc_getContextGraphInfoUpdateableTensorInfoFn_t)(
    QnnSystemDlc_ContextGraphInfoHandle_t handle, uint32_t index,
    QnnSystemDlc_TensorInfoHandle_t* tensorInfoHandle);

//
// From QnnSystemProfile.h
//

/** @brief See QnnSystemProfile_createSerializationTarget()*/
typedef Qnn_ErrorHandle_t (*QnnSystemProfile_createSerializationTargetFn_t)(QnnSystemProfile_SerializationTarget_t serializationTargetInfo,
                                                         QnnSystemProfile_SerializationTargetConfig_t* configs,
                                                         uint32_t numConfigs,
                                                         QnnSystemProfile_SerializationTargetHandle_t* serializationTarget);

/** @brief See QnnSystemProfile_serializeEventData()*/
typedef Qnn_ErrorHandle_t (*QnnSystemProfile_serializeEventDataFn_t)(QnnSystemProfile_SerializationTargetHandle_t serializationTarget,
                                                                 const QnnSystemProfile_ProfileData_t** eventData,
                                                                 uint32_t numEvents);

/** @brief See QnnSystemProfile_freeSerializationTarget()*/
typedef Qnn_ErrorHandle_t (*QnnSystemProfile_freeSerializationTargetFn_t)(QnnSystemProfile_SerializationTargetHandle_t serializationTarget);

/** @brief See QnnSystemDlc_freeRecordRange()*/
typedef Qnn_ErrorHandle_t (*QnnSystemDlc_freeRecordRangeFn_t)(QnnSystemDlc_RecordHandle_t recordHandle,
                                                              uint64_t offset,
                                                              uint64_t size);

/**
 * @brief This struct defines Qnn system interface specific to version.
 *        Interface functions are allowed to be NULL if not supported/available.
 *
 */
typedef struct {
  QnnSystemContext_CreateFn_t                    systemContextCreate;
  QnnSystemContext_GetBinaryInfoFn_t             systemContextGetBinaryInfo;
  QnnSystemContext_GetMetaDataFn_t               systemContextGetMetaData;
  QnnSystemContext_FreeFn_t                      systemContextFree;
  QnnSystemTensor_getMemoryFootprintFn_t         systemTensorGetMemoryFootprint;
  QnnSystemLog_createFn_t                        systemLogCreate;
  QnnSystemLog_setLogLevelFn_t                   systemLogSetLogLevel;
  QnnSystemLog_freeFn_t                          systemLogFree;
  QnnSystemDlc_createFromFileFn_t                systemDlcCreateFromFile;
  QnnSystemDlc_createFromBinaryFn_t              systemDlcCreateFromBinary;
  QnnSystemDlc_composeGraphsFn_t                 systemDlcComposeGraphs;
  QnnSystemDlc_getOpMappingsFn_t                 systemDlcGetOpMappings;
  QnnSystemDlc_freeFn_t                          systemDlcFree;
  QnnSystemProfile_createSerializationTargetFn_t systemProfileCreateSerializationTarget;
  QnnSystemProfile_serializeEventDataFn_t        systemProfileSerializeEventData;
  QnnSystemProfile_freeSerializationTargetFn_t   systemProfileFreeSerializationTarget;
  QnnSystemDlc_getRecordByNameFn_t               systemDlcGetRecordByName;
  QnnSystemDlc_getRecordsByTypeFn_t              systemDlcGetRecordsByType;
  QnnSystemDlc_getRecordDataSizeFn_t             systemDlcGetRecordDataSize;
  QnnSystemDlc_readRecordDataMemoryMappedFn_t    systemDlcReadRecordDataMemoryMapped;
  QnnSystemDlc_freeRecordFn_t                    systemDlcFreeRecord;
  QnnSystemDlc_saveFn_t                          systemDlcSave;
  QnnSystemDlc_createFromFileWithDestinationDirFn_t    systemDlcCreateFromFileWithDestinationDir;
  QnnSystemDlc_createFromBinaryWithDestinationDirFn_t  systemDlcCreateFromBinaryWithDestinationDir;
  QnnSystemDlc_getBinarySizeFn_t                       systemDlcGetBinarySize;
  QnnSystemDlc_getBinaryFn_t                           systemDlcGetBinary;
  QnnSystemDlc_createWithDestinationDirFn_t            systemDlcCreateWithDestinationDir;
  QnnSystemDlc_getComposableGraphInfoHandlesFn_t       systemDlcGetComposableGraphInfoHandles;
  QnnSystemDlc_getContextBinaryInfoHandlesFn_t         systemDlcGetContextBinaryInfoHandles;
  QnnSystemDlc_getComposableGraphInfoFn_t              systemDlcGetComposableGraphInfo;
  QnnSystemDlc_getContextBinaryInfoFn_t                systemDlcGetContextBinaryInfo;
  QnnSystemDlc_getComposableGraphInfoGraphNameFn_t     systemDlcGetComposableGraphInfoGraphName;
  QnnSystemDlc_getComposableGraphInfoNumInputTensorsFn_t  systemDlcGetComposableGraphInfoNumInputTensors;
  QnnSystemDlc_getComposableGraphInfoNumOutputTensorsFn_t systemDlcGetComposableGraphInfoNumOutputTensors;
  QnnSystemDlc_getGraphInfoInputTensorInfoFn_t   systemDlcGetGraphInfoInputTensorInfo;
  QnnSystemDlc_getGraphInfoOutputTensorInfoFn_t  systemDlcGetGraphInfoOutputTensorInfo;
  QnnSystemDlc_getTensorInfoNameFn_t                       systemDlcGetTensorInfoName;
  QnnSystemDlc_getTensorInfoIdFn_t                         systemDlcGetTensorInfoId;
  QnnSystemDlc_getTensorInfoTypeFn_t                       systemDlcGetTensorInfoType;
  QnnSystemDlc_getTensorInfoDataFormatFn_t                 systemDlcGetTensorInfoDataFormat;
  QnnSystemDlc_getTensorInfoDataTypeFn_t                   systemDlcGetTensorInfoDataType;
  QnnSystemDlc_getTensorInfoRankFn_t                       systemDlcGetTensorInfoRank;
  QnnSystemDlc_getTensorInfoDimensionsFn_t                 systemDlcGetTensorInfoDimensions;
  QnnSystemDlc_getTensorInfoQuantParamsFn_t                systemDlcGetTensorInfoQuantParams;
  QnnSystemDlc_getQuantParamsEncodingDefinitionFn_t    systemDlcGetQuantParamsEncodingDefinition;
  QnnSystemDlc_getQuantParamsEncodingFn_t              systemDlcGetQuantParamsEncoding;
  QnnSystemDlc_getQuantParamsScaleOffsetScaleFn_t      systemDlcGetQuantParamsScaleOffsetScale;
  QnnSystemDlc_getQuantParamsScaleOffsetOffsetFn_t     systemDlcGetQuantParamsScaleOffsetOffset;
  QnnSystemDlc_getQuantParamsBwScaleOffsetBitwidthFn_t systemDlcGetQuantParamsBwScaleOffsetBitwidth;
  QnnSystemDlc_getQuantParamsBwScaleOffsetScaleFn_t    systemDlcGetQuantParamsBwScaleOffsetScale;
  QnnSystemDlc_getQuantParamsBwScaleOffsetOffsetFn_t   systemDlcGetQuantParamsBwScaleOffsetOffset;
  QnnSystemDlc_getQuantParamsAxisScaleOffsetAxisFn_t   systemDlcGetQuantParamsAxisScaleOffsetAxis;
  QnnSystemDlc_getQuantParamsAxisScaleOffsetNumScaleOffsetsFn_t systemDlcGetQuantParamsAxisScaleOffsetNumScaleOffsets;
  QnnSystemDlc_getQuantParamsAxisScaleOffsetScaleAtIndexFn_t    systemDlcGetQuantParamsAxisScaleOffsetScaleAtIndex;
  QnnSystemDlc_getQuantParamsAxisScaleOffsetOffsetAtIndexFn_t   systemDlcGetQuantParamsAxisScaleOffsetOffsetAtIndex;
  QnnSystemDlc_getQuantParamsBwAxisScaleOffsetBitwidthFn_t      systemDlcGetQuantParamsBwAxisScaleOffsetBitwidth;
  QnnSystemDlc_getQuantParamsBwAxisScaleOffsetAxisFn_t          systemDlcGetQuantParamsBwAxisScaleOffsetAxis;
  QnnSystemDlc_getQuantParamsBwAxisScaleOffsetNumElementsFn_t   systemDlcGetQuantParamsBwAxisScaleOffsetNumElements;
  QnnSystemDlc_getQuantParamsBwAxisScaleOffsetScaleAtIndexFn_t  systemDlcGetQuantParamsBwAxisScaleOffsetScaleAtIndex;
  QnnSystemDlc_getQuantParamsBwAxisScaleOffsetOffsetAtIndexFn_t systemDlcGetQuantParamsBwAxisScaleOffsetOffsetAtIndex;
  QnnSystemDlc_getContextBinaryInfoBackendIdFn_t                systemDlcGetContextBinaryInfoBackendId;
  QnnSystemDlc_getContextBinaryInfoBuildIdFn_t                  systemDlcGetContextBinaryInfoBuildId;
  QnnSystemDlc_getContextBinaryInfoCoreApiVersionFn_t           systemDlcGetContextBinaryInfoCoreApiVersion;
  QnnSystemDlc_getContextBinaryInfoBackendApiVersionFn_t        systemDlcGetContextBinaryInfoBackendApiVersion;
  QnnSystemDlc_getContextBinaryInfoSocVersionFn_t               systemDlcGetContextBinaryInfoSocVersion;
  QnnSystemDlc_getContextBinaryInfoHwInfoBlobVersionFn_t        systemDlcGetContextBinaryInfoHwInfoBlobVersion;
  QnnSystemDlc_getContextBinaryInfoContextBlobVersionFn_t       systemDlcGetContextBinaryInfoContextBlobVersion;
  QnnSystemDlc_getContextBinaryInfoHwInfoBlobSizeFn_t           systemDlcGetContextBinaryInfoHwInfoBlobSize;
  QnnSystemDlc_getContextBinaryInfoHwInfoBlobFn_t               systemDlcGetContextBinaryInfoHwInfoBlob;
  QnnSystemDlc_getContextBinaryInfoContextBlobSizeFn_t          systemDlcGetContextBinaryInfoContextBlobSize;
  QnnSystemDlc_getContextBinaryInfoNumContextTensorsFn_t        systemDlcGetContextBinaryInfoNumContextTensors;
  QnnSystemDlc_getContextBinaryInfoContextTensorInfoFn_t        systemDlcGetContextBinaryInfoContextTensorInfo;
  QnnSystemDlc_getContextBinaryInfoNumGraphsFn_t                systemDlcGetContextBinaryInfoNumGraphs;
  QnnSystemDlc_getContextGraphInfoHandleFn_t                    systemDlcGetContextGraphInfoHandle;
  QnnSystemDlc_getContextBinaryInfoSocModelFn_t                 systemDlcGetContextBinaryInfoSocModel;
  QnnSystemDlc_getContextBinaryInfoContextMetadataSizeFn_t      systemDlcGetContextBinaryInfoContextMetadataSize;
  QnnSystemDlc_getContextBinaryInfoContextMetadataFn_t          systemDlcGetContextBinaryInfoContextMetadata;
  QnnSystemDlc_getContextGraphInfoGraphNameFn_t                 systemDlcGetContextGraphInfoGraphName;
  QnnSystemDlc_getContextGraphInfoNumInputTensorsFn_t           systemDlcGetContextGraphInfoNumInputTensors;
  QnnSystemDlc_getContextGraphInfoNumOutputTensorsFn_t          systemDlcGetContextGraphInfoNumOutputTensors;
  QnnSystemDlc_getContextGraphInfoNumUpdateableTensorsFn_t      systemDlcGetContextGraphInfoNumUpdateableTensors;
  QnnSystemDlc_getContextGraphInfoGraphBlobInfoSizeFn_t         systemDlcGetContextGraphInfoGraphBlobInfoSize;
  QnnSystemDlc_getContextGraphInfoGraphBlobInfoFn_t             systemDlcGetContextGraphInfoGraphBlobInfo;
  QnnSystemDlc_getContextGraphInfoStartOpIndexFn_t              systemDlcGetContextGraphInfoStartOpIndex;
  QnnSystemDlc_getContextGraphInfoEndOpIndexFn_t                systemDlcGetContextGraphInfoEndOpIndex;
  QnnSystemDlc_getContextGraphInfoInputTensorInfoFn_t           systemDlcGetContextGraphInfoInputTensorInfo;
  QnnSystemDlc_getContextGraphInfoOutputTensorInfoFn_t          systemDlcGetContextGraphInfoOutputTensorInfo;
  QnnSystemDlc_getContextGraphInfoUpdateableTensorInfoFn_t      systemDlcGetContextGraphInfoUpdateableTensorInfo;
  QnnSystemDlc_freeRecordRangeFn_t                              systemDlcFreeRecordRange;
} QNN_SYSTEM_INTERFACE_VER_TYPE;

/// QNN_INTERFACE_VER_TYPE initializer macro
#define QNN_SYSTEM_INTERFACE_VER_TYPE_INIT { \
  NULL, /*systemContextCreate*/ \
  NULL, /*systemContextGetBinaryInfo*/ \
  NULL, /*systemContextGetMetaData*/ \
  NULL, /*systemContextFree*/ \
  NULL, /*systemTensorGetMemoryFootprint*/ \
  NULL, /*systemLogCreate*/ \
  NULL, /*systemLogSetLogLevel*/ \
  NULL, /*systemLogFree*/ \
  NULL, /*systemDlcCreateFromFile*/ \
  NULL, /*systemDlcCreateFromBinary*/ \
  NULL, /*systemDlcComposeGraphs*/ \
  NULL, /*systemDlcGetOpMappings*/ \
  NULL, /*systemDlcFree*/ \
  NULL, /*systemProfileCreateSerializationTarget*/ \
  NULL, /*systemProfileSerializeEventData*/ \
  NULL, /*systemProfileFreeSerializationTarget*/ \
  NULL, /*systemDlcGetRecordByName*/ \
  NULL, /*systemDlcGetRecordsByType*/ \
  NULL, /*systemDlcGetRecordDataSize*/ \
  NULL, /*systemDlcReadRecordDataMemoryMapped*/ \
  NULL, /*systemDlcFreeRecord*/ \
  NULL, /*systemDlcSave*/ \
  NULL, /*systemDlcCreateFromFileWithDestinationDir*/ \
  NULL, /*systemDlcCreateFromBinaryWithDestinationDir*/ \
  NULL, /*systemDlcGetBinarySize*/ \
  NULL, /*systemDlcGetBinary*/ \
  NULL, /*systemDlcCreateWithDestinationDir*/ \
  NULL, /*systemDlcGetComposableGraphInfoHandles*/ \
  NULL, /*systemDlcGetContextBinaryInfoHandles*/ \
  NULL, /*systemDlcGetComposableGraphInfo*/ \
  NULL, /*systemDlcGetContextBinaryInfo*/ \
  NULL, /*systemDlcGetComposableGraphInfoGraphName*/ \
  NULL, /*systemDlcGetComposableGraphInfoNumInputTensors*/ \
  NULL, /*systemDlcGetComposableGraphInfoNumOutputTensors*/ \
  NULL, /*systemDlcGetGraphInfoInputTensorInfo*/ \
  NULL, /*systemDlcGetGraphInfoOutputTensorInfo*/ \
  NULL, /*systemDlcGetTensorInfoName*/ \
  NULL, /*systemDlcGetTensorInfoId*/ \
  NULL, /*systemDlcGetTensorInfoType*/ \
  NULL, /*systemDlcGetTensorInfoDataFormat*/ \
  NULL, /*systemDlcGetTensorInfoDataType*/ \
  NULL, /*systemDlcGetTensorInfoRank*/ \
  NULL, /*systemDlcGetTensorInfoDimensions*/ \
  NULL, /*systemDlcGetTensorInfoQuantParams*/ \
  NULL, /*systemDlcGetQuantParamsEncodingDefinition*/ \
  NULL, /*systemDlcGetQuantParamsEncoding*/ \
  NULL, /*systemDlcGetQuantParamsScaleOffsetScale*/ \
  NULL, /*systemDlcGetQuantParamsScaleOffsetOffset*/ \
  NULL, /*systemDlcGetQuantParamsBwScaleOffsetBitwidth*/ \
  NULL, /*systemDlcGetQuantParamsBwScaleOffsetScale*/ \
  NULL, /*systemDlcGetQuantParamsBwScaleOffsetOffset*/ \
  NULL, /*systemDlcGetQuantParamsAxisScaleOffsetAxis*/ \
  NULL, /*systemDlcGetQuantParamsAxisScaleOffsetNumScaleOffsets*/ \
  NULL, /*systemDlcGetQuantParamsAxisScaleOffsetScaleAtIndex*/ \
  NULL, /*systemDlcGetQuantParamsAxisScaleOffsetOffsetAtIndex*/ \
  NULL, /*systemDlcGetQuantParamsBwAxisScaleOffsetBitwidth*/ \
  NULL, /*systemDlcGetQuantParamsBwAxisScaleOffsetAxis*/ \
  NULL, /*systemDlcGetQuantParamsBwAxisScaleOffsetNumElements*/ \
  NULL, /*systemDlcGetQuantParamsBwAxisScaleOffsetScaleAtIndex*/ \
  NULL, /*systemDlcGetQuantParamsBwAxisScaleOffsetOffsetAtIndex*/ \
  NULL, /*systemDlcGetContextBinaryInfoBackendId*/ \
  NULL, /*systemDlcGetContextBinaryInfoBuildId*/ \
  NULL, /*systemDlcGetContextBinaryInfoCoreApiVersion*/ \
  NULL, /*systemDlcGetContextBinaryInfoBackendApiVersion*/ \
  NULL, /*systemDlcGetContextBinaryInfoSocVersion*/ \
  NULL, /*systemDlcGetContextBinaryInfoHwInfoBlobVersion*/ \
  NULL, /*systemDlcGetContextBinaryInfoContextBlobVersion*/ \
  NULL, /*systemDlcGetContextBinaryInfoHwInfoBlobSize*/ \
  NULL, /*systemDlcGetContextBinaryInfoHwInfoBlob*/ \
  NULL, /*systemDlcGetContextBinaryInfoContextBlobSize*/ \
  NULL, /*systemDlcGetContextBinaryInfoNumContextTensors*/ \
  NULL, /*systemDlcGetContextBinaryInfoContextTensorInfo*/ \
  NULL, /*systemDlcGetContextBinaryInfoNumGraphs*/ \
  NULL, /*systemDlcGetContextGraphInfoHandle*/ \
  NULL, /*systemDlcGetContextBinaryInfoSocModel*/ \
  NULL, /*systemDlcGetContextBinaryInfoContextMetadataSize*/ \
  NULL, /*systemDlcGetContextBinaryInfoContextMetadata*/ \
  NULL, /*systemDlcGetContextGraphInfoGraphName*/ \
  NULL, /*systemDlcGetContextGraphInfoNumInputTensors*/ \
  NULL, /*systemDlcGetContextGraphInfoNumOutputTensors*/ \
  NULL, /*systemDlcGetContextGraphInfoNumUpdateableTensors*/ \
  NULL, /*systemDlcGetContextGraphInfoGraphBlobInfoSize*/ \
  NULL, /*systemDlcGetContextGraphInfoGraphBlobInfo*/ \
  NULL, /*systemDlcGetContextGraphInfoStartOpIndex*/ \
  NULL, /*systemDlcGetContextGraphInfoEndOpIndex*/ \
  NULL, /*systemDlcGetContextGraphInfoInputTensorInfo*/ \
  NULL, /*systemDlcGetContextGraphInfoOutputTensorInfo*/ \
  NULL, /*systemDlcGetContextGraphInfoUpdateableTensorInfo*/ \
  NULL, /*systemDlcFreeRecordRange*/ \
}

typedef struct {
  /// Backend identifier. See QnnCommon.h for details.
  /// Allowed to be QNN_BACKEND_ID_NULL in case of single backend library or a dedicated system
  /// library, in which case clients can deduce backend identifier based on library being loaded.
  uint32_t backendId;
  /// Interface provider name. Allowed to be NULL.
  const char* providerName;
  // API version for provided interface
  Qnn_Version_t systemApiVersion;
  union UNNAMED {
    // Core interface type and name: e.g. QnnSystemInterface_ImplementationV0_0_t v0_0;
    QNN_SYSTEM_INTERFACE_VER_TYPE  QNN_SYSTEM_INTERFACE_VER_NAME;
  };
} QnnSystemInterface_t;

/// QnnSystemInterface_t initializer macro
#define QNN_SYSTEM_INTERFACE_INIT                                          \
  {                                                                        \
    QNN_BACKEND_ID_NULL,     /*backendId*/                                 \
    NULL,                    /*providerName*/                              \
    QNN_VERSION_INIT,        /*apiVersion*/                                \
    {                                                                      \
      QNN_SYSTEM_INTERFACE_VER_TYPE_INIT /*QNN_SYSTEM_INTERFACE_VER_NAME*/ \
    }                                                                      \
  }

// clang-format on

//=============================================================================
// Public Functions
//=============================================================================

/**
 * @brief Get list of available interface providers.
 *
 * @param[out] providerList A pointer to an array of available interface providers.
 *                          The lifetime of returned interface object pointers
 *                          corresponds to the lifetime of the provider library.
 *                          Contents are to be considered invalid if the provider
 *                          library is terminated/unloaded.
 *                          This function can be called immediately after provider
 *                          library has been loaded.
 * @param[out] numProviders Number of available interface objects in _providerList_.
 *
 * @return Error code:
 *         - QNN_SUCCESS: No error.
 *         - QNN_SYSTEM_INTERFACE_INVALID_PARAMETER: Invalid parameter was provided.
 *           Either _providerList_ or _numProviders_ was NULL.
 *         - QNN_SYSTEM_INTERFACE_ERROR_NOT_SUPPORTED: API not supported.
 */
QNN_SYSTEM_INTERFACE
Qnn_ErrorHandle_t QnnSystemInterface_getProviders(const QnnSystemInterface_t*** providerList,
                                                  uint32_t* numProviders);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // QNN_SYSTEM_INTERFACE_H
