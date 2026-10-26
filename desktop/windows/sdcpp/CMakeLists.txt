cmake_minimum_required(VERSION 3.22)

# stable-diffusion.cpp for Windows on Arm (Snapdragon): sd-cli.exe with LTX-Video and the Hexagon NPU transformer and
# VAE (QNN context binaries, SD_LTXV_QNN). The same sources as the Android app: android/stable-diffusion.cpp on the
# ggml vendored in android/llama.cpp, QNN headers from android/sdcpp. Built by build.ps1 with llvm-mingw
# (aarch64-w64-mingw32); QnnHtp.dll / QnnSystem.dll are loaded at run time from the app folder.
project(llmhub_sdcpp_windows LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(REPO_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../..")
set(ANDROID_ROOT "${REPO_ROOT}/android")

# One self-contained executable (no libc++ / libunwind DLLs next to it).
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static")

add_compile_definitions(GGML_MAX_NAME=160)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
# Snapdragon X (Oryon) supports dotprod, fp16 and i8mm.
set(GGML_CPU_ARM_ARCH "armv8.2-a+dotprod+fp16+i8mm" CACHE STRING "" FORCE)
set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
set(GGML_CPU_ALL_VARIANTS OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_LLAMAFILE OFF CACHE BOOL "" FORCE)
set(GGML_KLEIDIAI OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

set(GGML_SOURCE_DIR "${ANDROID_ROOT}/llama.cpp/ggml")
add_subdirectory("${GGML_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/ggml")

set(SD_USE_UPSTREAM_GGML ON CACHE BOOL "" FORCE)
set(SD_BUILD_EXAMPLES ON CACHE BOOL "" FORCE)
set(SD_WEBP OFF CACHE BOOL "" FORCE)
set(SD_WEBM OFF CACHE BOOL "" FORCE)
set(SD_GGML_SOURCE_DIR "${GGML_SOURCE_DIR}" CACHE PATH "" FORCE)
add_subdirectory("${ANDROID_ROOT}/stable-diffusion.cpp" "${CMAKE_CURRENT_BINARY_DIR}/stable-diffusion.cpp")
target_compile_definitions(stable-diffusion PRIVATE SD_LTXV_QNN=1)
target_include_directories(stable-diffusion PRIVATE "${ANDROID_ROOT}/sdcpp/src/main/cpp/qnn/include/QNN")
