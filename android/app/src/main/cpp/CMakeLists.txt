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

set(LLAMA_CPP_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../../llama.cpp")
add_subdirectory("${LLAMA_CPP_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/llama.cpp")

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

# The official b11179 Snapdragon release is packaged alongside the app, not fetched
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
