cmake_minimum_required(VERSION 3.22.1)
project(llm_hub_jni C CXX)

# This is the primary build script for the JNI library.
# The llama.cpp project is included as a sub-project.
#
# IMPORTANT:
# The `set(VARIABLE ... CACHE ...)` commands below are used to
# override default settings in the llama.cpp sub-project. They
# MUST appear before the `add_subdirectory` call that includes
# the project.
# -----------------------------------------------------------------

# Disable "native" CPU optimizations because they add -mcpu=native which is
# not supported by the Android NDK cross-compiler.
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
set(LLAMA_NATIVE OFF CACHE BOOL "" FORCE)

# Enable OpenCL backend so we can utilize GPU via OpenCL
set(GGML_OPENCL ON CACHE BOOL "" FORCE)

# Disable embedding kernels to avoid Python requirement
set(GGML_OPENCL_EMBED_KERNELS OFF CACHE BOOL "" FORCE)

# -----------------------------------------------------------------
# Provide OpenCL headers and an ICD loader implementation so that
# linking succeeds even though the Android NDK does not ship a stub
# libOpenCL.so.  We use the official Khronos OpenCL-ICD-Loader which
# will forward the calls to the vendor implementation on the device
# at runtime (via dlopen of the vendor library).

# Fetch OpenCL headers (if not already present)
include(FetchContent)
FetchContent_Declare(
    opencl_headers
    GIT_REPOSITORY https://github.com/KhronosGroup/OpenCL-Headers.git
    GIT_TAG        v2024.05.08
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(opencl_headers)

# Fetch and build the ICD loader
FetchContent_Declare(
    opencl_loader
    GIT_REPOSITORY https://github.com/KhronosGroup/OpenCL-ICD-Loader.git
    GIT_TAG        v2024.05.08
    GIT_SHALLOW    TRUE
)

# Configure the loader: build a shared library named libOpenCL.so
set(OPENCL_ICD_LOADER_HEADERS_DIR ${opencl_headers_SOURCE_DIR}/include CACHE PATH "")
set(BUILD_ICD_LOADER ON CACHE BOOL "" FORCE)
set(OPENCL_ICD_LOADER_BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(opencl_loader)

# The loader project defines the target OpenCL::OpenCL; expose variables for ggml
set(OpenCL_INCLUDE_DIR "${opencl_headers_SOURCE_DIR}/include" CACHE PATH "")
set(OpenCL_INCLUDE_DIRS "${opencl_headers_SOURCE_DIR}/include" CACHE PATH "")
set(OpenCL_LIBRARY OpenCL::OpenCL CACHE STRING "" FORCE)
set(OpenCL_LIBRARIES OpenCL::OpenCL CACHE STRING "" FORCE)
set(OpenCL_FOUND TRUE CACHE BOOL "" FORCE)

# Add llama.cpp as a subdirectory. This assumes it's available as a
# git submodule at ../../../../llama.cpp relative to this CMakeLists.txt
add_subdirectory(../../../../llama.cpp ${CMAKE_CURRENT_BINARY_DIR}/llama.cpp)

# Define the JNI library
add_library(
        llm-hub-jni
        SHARED
        llm-hub-jni.cpp)

# Add include directories for llama.h and ggml.h
target_include_directories(llm-hub-jni PRIVATE
        ${llama_SOURCE_DIR}/include
        ${llama_SOURCE_DIR}/ggml/include
        )

# Link against the llama library
target_link_libraries(
        llm-hub-jni
        llama
        android
        log
        ) 