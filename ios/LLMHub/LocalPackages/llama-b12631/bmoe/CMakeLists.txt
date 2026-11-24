# Static library for the iOS app's flash streaming: BigMoeOnEdge's engine (android/bigmoeonedge/core), llama.cpp's
# common helpers it needs, and the C bridge (llmhub_bmoe.cpp). llama, ggml and mtmd themselves come from the app's
# llama.xcframework at link time, so they are built here only to satisfy CMake and are not packaged.
cmake_minimum_required(VERSION 3.22)
project(llmhub_bmoe LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(LLAMA_SOURCE_DIR "" CACHE PATH "llama.cpp source the xcframework was built from (b12631 + llmhub-aion.patch)")
set(BMOE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../../../android/bigmoeonedge")
if(NOT EXISTS "${LLAMA_SOURCE_DIR}/include/llama.h")
    message(FATAL_ERROR "Set LLAMA_SOURCE_DIR to the llama.cpp source")
endif()

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)
set(LLAMA_BUILD_MTMD ON CACHE BOOL "" FORCE)
set(MTMD_VIDEO OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP OFF CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
set(GGML_METAL OFF CACHE BOOL "" FORCE)
set(GGML_BLAS OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
add_subdirectory("${LLAMA_SOURCE_DIR}" llama.cpp EXCLUDE_FROM_ALL)

file(STRINGS "${BMOE_DIR}/LLMHUB_VERSION.md" BMOE_VERSION_LINE REGEX "^- Version: ")
string(REGEX MATCH "[0-9]+\\.[0-9]+\\.[0-9]+" PROJECT_VERSION "${BMOE_VERSION_LINE}")
set(BMOE_HAVE_LLAMA ON)
add_subdirectory("${BMOE_DIR}/core" bigmoeonedge EXCLUDE_FROM_ALL)

add_library(llmhub_bmoe STATIC llmhub_bmoe.cpp)
target_include_directories(llmhub_bmoe PUBLIC include)
target_link_libraries(llmhub_bmoe PRIVATE bmoe_core)
