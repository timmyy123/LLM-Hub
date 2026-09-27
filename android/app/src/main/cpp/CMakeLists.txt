cmake_minimum_required(VERSION 3.22.1)

project(llmhub_llama LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Keep the baseline ARMv8 CPU runtime for devices without Snapdragon OpenCL/Hexagon.
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_MTMD ON CACHE BOOL "" FORCE)
set(MTMD_VIDEO OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
set(GGML_CPU_ARM_ARCH "armv8-a" CACHE STRING "" FORCE)
set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
set(GGML_CPU_ALL_VARIANTS OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_LLAMAFILE OFF CACHE BOOL "" FORCE)
set(GGML_KLEIDIAI OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN ON CACHE BOOL "" FORCE)

# The NDK ships Android Vulkan headers, SPIR-V headers and a host glslc. Keep
# Vulkan builds self-contained in Android Studio without a separate Vulkan SDK.
if(NOT CMAKE_ANDROID_NDK)
    message(FATAL_ERROR "Android NDK is required for the GGUF Vulkan backend")
endif()
file(GLOB NDK_GLSLC_CANDIDATES "${CMAKE_ANDROID_NDK}/shader-tools/*/glslc" "${CMAKE_ANDROID_NDK}/shader-tools/*/glslc.exe")
list(LENGTH NDK_GLSLC_CANDIDATES NDK_GLSLC_COUNT)
if(NDK_GLSLC_COUNT EQUAL 0)
    message(FATAL_ERROR "NDK host glslc was not found")
endif()
list(GET NDK_GLSLC_CANDIDATES 0 NDK_GLSLC)
set(Vulkan_GLSLC_EXECUTABLE "${NDK_GLSLC}" CACHE FILEPATH "" FORCE)
set(Vulkan_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/vulkan-headers" CACHE PATH "" FORCE)
set(SPIRV-Headers_DIR "${CMAKE_CURRENT_LIST_DIR}/cmake" CACHE PATH "" FORCE)
set(NDK_SPIRV_HEADERS "${CMAKE_ANDROID_NDK}/sources/third_party/shaderc/third_party/spirv-tools/external/spirv-headers/include")

set(LLAMA_CPP_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../../llama.cpp")
add_subdirectory("${LLAMA_CPP_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/llama.cpp")
target_include_directories(ggml-vulkan PRIVATE "${NDK_SPIRV_HEADERS}")

add_library(llmhub_llama_cpu SHARED llama_cpp_jni.cpp)
target_compile_options(llmhub_llama_cpu PRIVATE
    $<$<COMPILE_LANGUAGE:C>:-march=armv8-a>
    $<$<COMPILE_LANGUAGE:CXX>:-march=armv8-a>
    -fvisibility=hidden
)
target_link_options(llmhub_llama_cpu PRIVATE
    "-Wl,--exclude-libs,ALL"
    "-Wl,-Bsymbolic"
)
target_include_directories(llmhub_llama_cpu PRIVATE
    "${LLAMA_CPP_DIR}/include"
    "${LLAMA_CPP_DIR}/ggml/include"
    "${LLAMA_CPP_DIR}/tools/mtmd"
)
target_link_libraries(llmhub_llama_cpu PRIVATE llama mtmd android log)

# Portable arm64 llama.cpp source with the Vulkan GPU backend for Mali and other
# Android Vulkan devices. This is separate from the Snapdragon release binary.
add_library(llmhub_llama_vulkan SHARED llama_cpp_jni.cpp)
target_compile_definitions(llmhub_llama_vulkan PRIVATE LLMHUB_VULKAN=1)
target_compile_options(llmhub_llama_vulkan PRIVATE -march=armv8-a -fvisibility=hidden)
target_link_options(llmhub_llama_vulkan PRIVATE "-Wl,--exclude-libs,ALL" "-Wl,-Bsymbolic")
target_include_directories(llmhub_llama_vulkan PRIVATE
    "${LLAMA_CPP_DIR}/include"
    "${LLAMA_CPP_DIR}/ggml/include"
    "${LLAMA_CPP_DIR}/tools/mtmd")
target_link_libraries(llmhub_llama_vulkan PRIVATE llama mtmd android log)

# The official b11218 Snapdragon release is packaged alongside the app, not fetched
# by Gradle. It is loaded only when the user selects Adreno GPU or Hexagon NPU.
set(LLAMA_PREBUILT_LIB_DIR "${CMAKE_CURRENT_LIST_DIR}/../jniLibs/arm64-v8a")
foreach(prebuilt IN ITEMS ggml llama mtmd)
    add_library(llmhub_prebuilt_${prebuilt} SHARED IMPORTED)
    set_target_properties(llmhub_prebuilt_${prebuilt} PROPERTIES
        IMPORTED_LOCATION "${LLAMA_PREBUILT_LIB_DIR}/lib${prebuilt}.so")
endforeach()

add_library(llmhub_llama_snapdragon SHARED llama_cpp_jni.cpp)
target_compile_definitions(llmhub_llama_snapdragon PRIVATE LLMHUB_SNAPDRAGON=1)
target_include_directories(llmhub_llama_snapdragon PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/llama-prebuilt/include")
target_link_libraries(llmhub_llama_snapdragon PRIVATE
    llmhub_prebuilt_llama llmhub_prebuilt_mtmd llmhub_prebuilt_ggml android log)
target_link_options(llmhub_llama_snapdragon PRIVATE "-Wl,--allow-shlib-undefined")
