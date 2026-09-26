#include "MagentaLiteRTBridge.h"

#include <cstring>
#include <memory>
#include <vector>

namespace {

using LiteRtStatus = int32_t;
using LiteRtParamIndex = size_t;
using LiteRtEnvironment = void *;
using LiteRtModel = void *;
using LiteRtSignature = void *;
using LiteRtTensor = void *;
using LiteRtCompiledModel = void *;
using LiteRtOptions = void *;
using LiteRtTensorBuffer = void *;
using LiteRtTensorBufferRequirements = void *;

enum LiteRtElementType : int32_t {
  kLiteRtElementTypeFloat32 = 1,
  kLiteRtElementTypeInt32 = 2,
};

struct LiteRtLayout {
  unsigned int rank : 7;
  unsigned int has_strides : 1;
  int32_t dimensions[8];
  uint32_t strides[8];
};

struct LiteRtRankedTensorType {
  LiteRtElementType element_type;
  LiteRtLayout layout;
};

enum LiteRtTensorBufferLockMode : int32_t {
  kLiteRtTensorBufferLockModeRead = 0,
  kLiteRtTensorBufferLockModeWrite = 1,
};

extern "C" {
LiteRtStatus LiteRtCreateEnvironment(int, const void *, LiteRtEnvironment *);
void LiteRtDestroyEnvironment(LiteRtEnvironment);
LiteRtStatus LiteRtCreateModelFromFile(LiteRtEnvironment, const char *,
                                       LiteRtModel *);
void LiteRtDestroyModel(LiteRtModel);
LiteRtStatus LiteRtGetModelSignature(LiteRtModel, LiteRtParamIndex,
                                     LiteRtSignature *);
LiteRtStatus LiteRtGetNumSignatureInputs(LiteRtSignature, LiteRtParamIndex *);
LiteRtStatus LiteRtGetNumSignatureOutputs(LiteRtSignature, LiteRtParamIndex *);
LiteRtStatus LiteRtGetSignatureInputTensorByIndex(LiteRtSignature,
                                                  LiteRtParamIndex,
                                                  LiteRtTensor *);
LiteRtStatus LiteRtGetSignatureOutputTensorByIndex(LiteRtSignature,
                                                   LiteRtParamIndex,
                                                   LiteRtTensor *);
LiteRtStatus LiteRtGetRankedTensorType(LiteRtTensor,
                                       LiteRtRankedTensorType *);
LiteRtStatus LiteRtCreateCompiledModel(LiteRtEnvironment, LiteRtModel,
                                       LiteRtOptions, LiteRtCompiledModel *);
LiteRtStatus LiteRtCreateOptions(LiteRtOptions *);
void LiteRtDestroyOptions(LiteRtOptions);
LiteRtStatus LiteRtSetOptionsHardwareAccelerators(LiteRtOptions, int);
void LiteRtDestroyCompiledModel(LiteRtCompiledModel);
LiteRtStatus LiteRtGetCompiledModelInputBufferRequirements(
    LiteRtCompiledModel, LiteRtParamIndex, LiteRtParamIndex,
    LiteRtTensorBufferRequirements *);
LiteRtStatus LiteRtGetCompiledModelOutputBufferRequirements(
    LiteRtCompiledModel, LiteRtParamIndex, LiteRtParamIndex,
    LiteRtTensorBufferRequirements *);
LiteRtStatus LiteRtCreateManagedTensorBufferFromRequirements(
    LiteRtEnvironment, const LiteRtRankedTensorType *,
    LiteRtTensorBufferRequirements, LiteRtTensorBuffer *);
void LiteRtDestroyTensorBuffer(LiteRtTensorBuffer);
LiteRtStatus LiteRtGetTensorBufferPackedSize(LiteRtTensorBuffer, size_t *);
LiteRtStatus LiteRtLockTensorBuffer(LiteRtTensorBuffer, void **,
                                    LiteRtTensorBufferLockMode);
LiteRtStatus LiteRtUnlockTensorBuffer(LiteRtTensorBuffer);
LiteRtStatus LiteRtRunCompiledModel(LiteRtCompiledModel, LiteRtParamIndex,
                                    size_t, LiteRtTensorBuffer *, size_t,
                                    LiteRtTensorBuffer *);
}

constexpr LiteRtStatus kOk = 0;
constexpr int kCpuAccelerator = 1 << 0;

struct Resources {
  LiteRtEnvironment environment = nullptr;
  LiteRtModel model = nullptr;
  LiteRtCompiledModel compiled = nullptr;
  LiteRtOptions options = nullptr;
  std::vector<LiteRtTensorBuffer> inputs;
  std::vector<LiteRtTensorBuffer> outputs;

  ~Resources() {
    for (auto buffer : outputs) LiteRtDestroyTensorBuffer(buffer);
    for (auto buffer : inputs) LiteRtDestroyTensorBuffer(buffer);
    if (compiled) LiteRtDestroyCompiledModel(compiled);
    if (options) LiteRtDestroyOptions(options);
    if (model) LiteRtDestroyModel(model);
    if (environment) LiteRtDestroyEnvironment(environment);
  }
};

struct EmbeddingResources : Resources {
  int32_t sequence_length = 0;
  int32_t dimension = 0;
};

}  // namespace

void *LiteRTEmbeddingCreate(const char *model_path, int32_t *status_out) {
  if (status_out) *status_out = -1;
  if (!model_path) return nullptr;
  auto resources = std::make_unique<EmbeddingResources>();
  LiteRtStatus status = LiteRtCreateEnvironment(0, nullptr, &resources->environment);
  if (status != kOk) { if (status_out) *status_out = 1000 + status; return nullptr; }
  status = LiteRtCreateModelFromFile(resources->environment, model_path,
                                     &resources->model);
  if (status != kOk) { if (status_out) *status_out = 1100 + status; return nullptr; }

  LiteRtSignature signature = nullptr;
  LiteRtParamIndex input_count = 0, output_count = 0;
  status = LiteRtGetModelSignature(resources->model, 0, &signature);
  if (status != kOk || !signature ||
      LiteRtGetNumSignatureInputs(signature, &input_count) != kOk ||
      LiteRtGetNumSignatureOutputs(signature, &output_count) != kOk ||
      input_count != 1 || output_count != 1) {
    if (status_out) *status_out = -2;
    return nullptr;
  }

  LiteRtTensor input_tensor = nullptr, output_tensor = nullptr;
  LiteRtRankedTensorType input_type{}, output_type{};
  if (LiteRtGetSignatureInputTensorByIndex(signature, 0, &input_tensor) != kOk ||
      LiteRtGetSignatureOutputTensorByIndex(signature, 0, &output_tensor) != kOk ||
      LiteRtGetRankedTensorType(input_tensor, &input_type) != kOk ||
      LiteRtGetRankedTensorType(output_tensor, &output_type) != kOk ||
      input_type.element_type != kLiteRtElementTypeInt32 ||
      output_type.element_type != kLiteRtElementTypeFloat32 ||
      input_type.layout.rank != 2 || output_type.layout.rank != 2 ||
      input_type.layout.dimensions[0] != 1 ||
      output_type.layout.dimensions[0] != 1) {
    if (status_out) *status_out = -3;
    return nullptr;
  }
  resources->sequence_length = input_type.layout.dimensions[1];
  resources->dimension = output_type.layout.dimensions[1];
  if (resources->sequence_length <= 0 || resources->dimension <= 0) {
    if (status_out) *status_out = -4;
    return nullptr;
  }

  status = LiteRtCreateOptions(&resources->options);
  if (status != kOk) { if (status_out) *status_out = 1200 + status; return nullptr; }
  status = LiteRtSetOptionsHardwareAccelerators(resources->options,
                                                kCpuAccelerator);
  if (status != kOk) { if (status_out) *status_out = 1300 + status; return nullptr; }
  status = LiteRtCreateCompiledModel(resources->environment, resources->model,
                                     resources->options, &resources->compiled);
  if (status != kOk) { if (status_out) *status_out = 1400 + status; return nullptr; }

  LiteRtTensorBufferRequirements input_requirements = nullptr;
  LiteRtTensorBufferRequirements output_requirements = nullptr;
  LiteRtTensorBuffer input_buffer = nullptr, output_buffer = nullptr;
  if (LiteRtGetCompiledModelInputBufferRequirements(
          resources->compiled, 0, 0, &input_requirements) != kOk ||
      LiteRtCreateManagedTensorBufferFromRequirements(
          resources->environment, &input_type, input_requirements,
          &input_buffer) != kOk || !input_buffer) {
    if (status_out) *status_out = -5;
    return nullptr;
  }
  resources->inputs.push_back(input_buffer);
  if (LiteRtGetCompiledModelOutputBufferRequirements(
          resources->compiled, 0, 0, &output_requirements) != kOk ||
      LiteRtCreateManagedTensorBufferFromRequirements(
          resources->environment, &output_type, output_requirements,
          &output_buffer) != kOk || !output_buffer) {
    if (status_out) *status_out = -6;
    return nullptr;
  }
  resources->outputs.push_back(output_buffer);
  if (status_out) *status_out = 0;
  return resources.release();
}

void LiteRTEmbeddingDestroy(void *handle) {
  delete static_cast<EmbeddingResources *>(handle);
}

int32_t LiteRTEmbeddingSequenceLength(void *handle) {
  auto *resources = static_cast<EmbeddingResources *>(handle);
  return resources ? resources->sequence_length : 0;
}

int32_t LiteRTEmbeddingDimension(void *handle) {
  auto *resources = static_cast<EmbeddingResources *>(handle);
  return resources ? resources->dimension : 0;
}

int32_t LiteRTEmbeddingRun(void *handle, const int32_t *tokens,
                          size_t token_count, float *output,
                          size_t output_count) {
  auto *resources = static_cast<EmbeddingResources *>(handle);
  if (!resources || !tokens || !output ||
      token_count != static_cast<size_t>(resources->sequence_length) ||
      output_count != static_cast<size_t>(resources->dimension)) return -1;
  size_t packed_size = 0;
  void *destination = nullptr;
  if (LiteRtGetTensorBufferPackedSize(resources->inputs[0], &packed_size) != kOk ||
      packed_size != token_count * sizeof(int32_t) ||
      LiteRtLockTensorBuffer(resources->inputs[0], &destination,
                             kLiteRtTensorBufferLockModeWrite) != kOk ||
      !destination) return -2;
  std::memcpy(destination, tokens, packed_size);
  if (LiteRtUnlockTensorBuffer(resources->inputs[0]) != kOk) return -3;

  LiteRtStatus status = LiteRtRunCompiledModel(
      resources->compiled, 0, resources->inputs.size(), resources->inputs.data(),
      resources->outputs.size(), resources->outputs.data());
  if (status != kOk) return 1500 + status;
  void *source = nullptr;
  if (LiteRtGetTensorBufferPackedSize(resources->outputs[0], &packed_size) != kOk ||
      packed_size != output_count * sizeof(float) ||
      LiteRtLockTensorBuffer(resources->outputs[0], &source,
                             kLiteRtTensorBufferLockModeRead) != kOk ||
      !source) return -4;
  std::memcpy(output, source, packed_size);
  if (LiteRtUnlockTensorBuffer(resources->outputs[0]) != kOk) return -5;
  return 0;
}

int32_t MagentaLiteRTRunModel(const char *model_path,
                              const int32_t *int32_input,
                              size_t int32_input_count,
                              const float *float_input,
                              size_t float_input_count,
                              const float *second_float_input,
                              size_t second_float_input_count, void *output,
                              size_t output_capacity,
                              int32_t expected_output_type) {
  if (!model_path || !output) return -1;

  Resources resources;
  LiteRtStatus status =
      LiteRtCreateEnvironment(0, nullptr, &resources.environment);
  if (status != kOk) return 1000 + status;
  status = LiteRtCreateModelFromFile(resources.environment, model_path,
                                     &resources.model);
  if (status != kOk) return 1100 + status;

  LiteRtSignature signature = nullptr;
  status = LiteRtGetModelSignature(resources.model, 0, &signature);
  if (status != kOk || !signature) return 1200 + status;

  LiteRtParamIndex input_count = 0;
  LiteRtParamIndex output_count = 0;
  if (LiteRtGetNumSignatureInputs(signature, &input_count) != kOk ||
      LiteRtGetNumSignatureOutputs(signature, &output_count) != kOk ||
      output_count < 1) {
    return -2;
  }
  // LiteRT 2.x requires a real options object. Passing nullptr returns
  // kLiteRtStatusErrorInvalidArgument (surfaced by Swift as status 1301).
  // MusicCoCa's prompt models are CPU graphs in the upstream Magenta runtime.
  status = LiteRtCreateOptions(&resources.options);
  if (status != kOk) return 1250 + status;
  status = LiteRtSetOptionsHardwareAccelerators(resources.options,
                                                kCpuAccelerator);
  if (status != kOk) return 1275 + status;
  status = LiteRtCreateCompiledModel(resources.environment, resources.model,
                                     resources.options, &resources.compiled);
  if (status != kOk) return 1300 + status;

  resources.inputs.reserve(input_count);
  for (LiteRtParamIndex index = 0; index < input_count; ++index) {
    LiteRtTensor tensor = nullptr;
    LiteRtRankedTensorType tensor_type{};
    LiteRtTensorBufferRequirements requirements = nullptr;
    LiteRtTensorBuffer buffer = nullptr;
    if (LiteRtGetSignatureInputTensorByIndex(signature, index, &tensor) != kOk ||
        LiteRtGetRankedTensorType(tensor, &tensor_type) != kOk ||
        LiteRtGetCompiledModelInputBufferRequirements(
            resources.compiled, 0, index, &requirements) != kOk ||
        LiteRtCreateManagedTensorBufferFromRequirements(
            resources.environment, &tensor_type, requirements, &buffer) != kOk) {
      return -3;
    }
    resources.inputs.push_back(buffer);

    size_t packed_size = 0;
    void *destination = nullptr;
    if (LiteRtGetTensorBufferPackedSize(buffer, &packed_size) != kOk ||
        LiteRtLockTensorBuffer(buffer, &destination,
                               kLiteRtTensorBufferLockModeWrite) != kOk ||
        !destination) {
      return -4;
    }

    const void *source = nullptr;
    size_t source_size = 0;
    if (tensor_type.element_type == kLiteRtElementTypeInt32 && int32_input) {
      source = int32_input;
      source_size = int32_input_count * sizeof(int32_t);
    } else if (tensor_type.element_type == kLiteRtElementTypeFloat32) {
      const bool use_second = index == 1 && second_float_input;
      source = use_second ? second_float_input : float_input;
      source_size = (use_second ? second_float_input_count : float_input_count) *
                    sizeof(float);
    }
    if (!source || source_size != packed_size) {
      LiteRtUnlockTensorBuffer(buffer);
      return -5;
    }
    std::memcpy(destination, source, source_size);
    if (LiteRtUnlockTensorBuffer(buffer) != kOk) return -6;
  }

  LiteRtTensor output_tensor = nullptr;
  LiteRtRankedTensorType output_type{};
  LiteRtTensorBufferRequirements output_requirements = nullptr;
  LiteRtTensorBuffer output_buffer = nullptr;
  if (LiteRtGetSignatureOutputTensorByIndex(signature, 0, &output_tensor) != kOk ||
      LiteRtGetRankedTensorType(output_tensor, &output_type) != kOk ||
      output_type.element_type != expected_output_type ||
      LiteRtGetCompiledModelOutputBufferRequirements(
          resources.compiled, 0, 0, &output_requirements) != kOk ||
      LiteRtCreateManagedTensorBufferFromRequirements(
          resources.environment, &output_type, output_requirements,
          &output_buffer) != kOk) {
    return -7;
  }
  resources.outputs.push_back(output_buffer);

  status = LiteRtRunCompiledModel(resources.compiled, 0, resources.inputs.size(),
                                  resources.inputs.data(), 1,
                                  resources.outputs.data());
  if (status != kOk) return 1400 + status;

  size_t packed_size = 0;
  void *source = nullptr;
  if (LiteRtGetTensorBufferPackedSize(output_buffer, &packed_size) != kOk ||
      packed_size > output_capacity ||
      LiteRtLockTensorBuffer(output_buffer, &source,
                             kLiteRtTensorBufferLockModeRead) != kOk ||
      !source) {
    return -8;
  }
  std::memcpy(output, source, packed_size);
  if (LiteRtUnlockTensorBuffer(output_buffer) != kOk) return -9;
  return 0;
}
