cmake_minimum_required(VERSION 3.10)
project(whisper_jni)

set(CMAKE_CXX_STANDARD 17)

# whisper.cpp source directory — set via WHISPER_CPP_DIR or default to submodule location
if(NOT DEFINED WHISPER_CPP_DIR)
    set(WHISPER_CPP_DIR "${CMAKE_SOURCE_DIR}/../../../../../whisper.cpp")
endif()

# Build whisper.cpp as a static library
set(WHISPER_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_CCACHE OFF CACHE BOOL "" FORCE)
# Force ARMv8.2-a+dotprod+fp16 — without this, cross-compile gets armv8-a baseline (no NEON extensions)
set(GGML_CPU_ARM_ARCH "armv8.2-a+dotprod+fp16" CACHE STRING "" FORCE)

add_subdirectory(${WHISPER_CPP_DIR} ${CMAKE_BINARY_DIR}/whisper.cpp)

# JNI bridge — named whisper_jni to avoid conflict with whisper.cpp's own "whisper" target
add_library(whisper_jni SHARED whisper_jni.cpp)

target_include_directories(whisper_jni PRIVATE
    ${WHISPER_CPP_DIR}/include
    ${WHISPER_CPP_DIR}/ggml/include
)

target_link_libraries(whisper_jni
    whisper
    android
    log
)
