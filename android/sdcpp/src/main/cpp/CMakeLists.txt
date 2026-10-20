cmake_minimum_required(VERSION 3.22.1)

project(llmhub_sdcpp LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(ANDROID_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../../..")
set(APP_CPP_DIR "${ANDROID_ROOT}/app/src/main/cpp")

# Own ggml build (the source is the vendored llama.cpp ggml). stable-diffusion.cpp needs 160-char
# tensor names; video generation requires >= 8 GB RAM devices, which all support dotprod/fp16.
add_compile_definitions(GGML_MAX_NAME=160)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
set(GGML_CPU_ARM_ARCH "armv8.2-a+dotprod+fp16" CACHE STRING "" FORCE)
set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
set(GGML_CPU_ALL_VARIANTS OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_LLAMAFILE OFF CACHE BOOL "" FORCE)
set(GGML_KLEIDIAI OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN ON CACHE BOOL "" FORCE)

# Same self-contained Vulkan setup as the app's llama.cpp build: NDK glslc + vendored headers.
if(NOT CMAKE_ANDROID_NDK)
    message(FATAL_ERROR "Android NDK is required for the ggml Vulkan backend")
endif()
file(GLOB NDK_GLSLC_CANDIDATES "${CMAKE_ANDROID_NDK}/shader-tools/*/glslc" "${CMAKE_ANDROID_NDK}/shader-tools/*/glslc.exe")
list(GET NDK_GLSLC_CANDIDATES 0 NDK_GLSLC)
set(Vulkan_GLSLC_EXECUTABLE "${NDK_GLSLC}" CACHE FILEPATH "" FORCE)
set(Vulkan_INCLUDE_DIR "${APP_CPP_DIR}/vulkan-headers" CACHE PATH "" FORCE)
set(SPIRV-Headers_DIR "${APP_CPP_DIR}/cmake" CACHE PATH "" FORCE)
set(NDK_SPIRV_HEADERS "${CMAKE_ANDROID_NDK}/sources/third_party/shaderc/third_party/spirv-tools/external/spirv-headers/include")

set(GGML_SOURCE_DIR "${ANDROID_ROOT}/llama.cpp/ggml")
add_subdirectory("${GGML_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/ggml")
target_include_directories(ggml-vulkan PRIVATE "${NDK_SPIRV_HEADERS}")

# stable-diffusion.cpp (vendored, with LTX-Video 0.9 + Hexagon NPU support) on top of that ggml.
set(SD_USE_UPSTREAM_GGML ON CACHE BOOL "" FORCE)
set(SD_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(SD_WEBP OFF CACHE BOOL "" FORCE)
set(SD_WEBM OFF CACHE BOOL "" FORCE)
set(SD_GGML_SOURCE_DIR "${GGML_SOURCE_DIR}" CACHE PATH "" FORCE)
add_subdirectory("${ANDROID_ROOT}/stable-diffusion.cpp" "${CMAKE_CURRENT_BINARY_DIR}/stable-diffusion.cpp")
# Hexagon NPU transformer/VAE (QNN context binaries); QNN libraries are dlopen'ed at runtime.
target_compile_definitions(stable-diffusion PRIVATE SD_LTXV_QNN=1)
target_include_directories(stable-diffusion PRIVATE "${CMAKE_CURRENT_LIST_DIR}/qnn/include/QNN")

add_library(llmhub_sdcpp SHARED sdcpp_jni.cpp imagegen_sdcpp_jni.cpp)
target_include_directories(llmhub_sdcpp PRIVATE "${ANDROID_ROOT}/stable-diffusion.cpp/thirdparty")
target_compile_options(llmhub_sdcpp PRIVATE -fvisibility=hidden)
target_link_options(llmhub_sdcpp PRIVATE "-Wl,--exclude-libs,ALL" "-Wl,-Bsymbolic")
target_link_libraries(llmhub_sdcpp PRIVATE stable-diffusion android log)
