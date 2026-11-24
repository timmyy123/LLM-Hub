cmake_minimum_required(VERSION 3.22.1)

project(llmhub_llama LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# llama.cpp is built as shared libraries so the CPU backend can be a set of per-chip modules
# (GGML_CPU_ALL_VARIANTS: ARMv8.0 up to ARMv9.2 with SVE2/SME) that the JNI library picks from at
# runtime. Every library from this tree is renamed to libllmhub_* below, because the packaged
# Snapdragon release ships libggml*.so/libllama.so/libmtmd.so under the upstream names and its
# ggml_backend_load_all_from_path() scan of the native library directory must not find ours.
set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
# llama-common is needed by the flash streaming runtime (bigmoeonedge) for chat parsing.
set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_MTMD ON CACHE BOOL "" FORCE)
set(MTMD_VIDEO OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
set(GGML_CPU_ARM_ARCH "" CACHE STRING "" FORCE)
set(GGML_BACKEND_DL ON CACHE BOOL "" FORCE)
set(GGML_CPU_ALL_VARIANTS ON CACHE BOOL "" FORCE)
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

set(LLAMA_CPP_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../../../llama.cpp")
# Every compile command carries the vendored llama.cpp version, so a version bump rebuilds all of
# llama.cpp/ggml. Upgrades unpack upstream archives whose file dates can be older than existing
# objects; without this, Ninja kept a stale llama-model.cpp.o across the b11738 bump, whose
# llama_model layout no longer matched the rest of the build (SIGSEGV in llama_vocab::n_tokens on
# every CPU model load in 4.5.2/4.5.3).
file(STRINGS "${LLAMA_CPP_DIR}/LLMHUB_VERSION.md" LLMHUB_LLAMA_VERSION_LINE REGEX "^- Version: ")
string(REGEX MATCH "b[0-9]+" LLMHUB_LLAMA_VERSION "${LLMHUB_LLAMA_VERSION_LINE}")
if(NOT LLMHUB_LLAMA_VERSION)
    message(FATAL_ERROR "No llama.cpp version in ${LLAMA_CPP_DIR}/LLMHUB_VERSION.md")
endif()
add_compile_definitions(LLMHUB_LLAMA_SOURCE_VERSION="${LLMHUB_LLAMA_VERSION}")
add_subdirectory("${LLAMA_CPP_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/llama.cpp")
# GGML_AVAILABLE_BACKENDS lists the backend modules: ggml-cpu-android_armv8.0_1 ... and ggml-vulkan.
foreach(llmhub_lib IN LISTS GGML_AVAILABLE_BACKENDS ITEMS ggml-base ggml llama llama-common mtmd)
    if(TARGET ${llmhub_lib})
        set_target_properties(${llmhub_lib} PROPERTIES OUTPUT_NAME "llmhub_${llmhub_lib}")
    endif()
endforeach()

# CPU runtime for every arm64 device. It loads the best libllmhub_ggml-cpu-* variant for the chip.
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
target_link_libraries(llmhub_llama_cpu PRIVATE llama mtmd ggml android log dl)

# The same llama.cpp libraries plus the libllmhub_ggml-vulkan module, for Mali and other Android
# Vulkan devices. This is separate from the Snapdragon release binary.
add_library(llmhub_llama_vulkan SHARED llama_cpp_jni.cpp)
target_compile_definitions(llmhub_llama_vulkan PRIVATE LLMHUB_VULKAN=1)
target_compile_options(llmhub_llama_vulkan PRIVATE -march=armv8-a -fvisibility=hidden)
target_link_options(llmhub_llama_vulkan PRIVATE "-Wl,--exclude-libs,ALL" "-Wl,-Bsymbolic")
target_include_directories(llmhub_llama_vulkan PRIVATE
    "${LLAMA_CPP_DIR}/include"
    "${LLAMA_CPP_DIR}/ggml/include"
    "${LLAMA_CPP_DIR}/tools/mtmd")
target_link_libraries(llmhub_llama_vulkan PRIVATE llama mtmd ggml android log dl)

# The official b12631 Snapdragon release is packaged alongside the app, not fetched
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

# Flash streaming runtime for MoE models: BigMoeOnEdge's engine, built on the llama.cpp libraries above (no fork).
# The routed experts are read from the model file on demand into a RAM cache, so a model larger than the
# phone's memory runs. CPU decode; the CPU variant comes from libllmhub_llama_cpu's nativeInit.
set(BMOE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../../../bigmoeonedge")
file(STRINGS "${BMOE_DIR}/LLMHUB_VERSION.md" BMOE_VERSION_LINE REGEX "^- Version: ")
string(REGEX MATCH "[0-9]+\\.[0-9]+\\.[0-9]+" BMOE_UPSTREAM_VERSION "${BMOE_VERSION_LINE}")
if(NOT BMOE_UPSTREAM_VERSION)
    message(FATAL_ERROR "No BigMoeOnEdge version in ${BMOE_DIR}/LLMHUB_VERSION.md")
endif()
set(PROJECT_VERSION "${BMOE_UPSTREAM_VERSION}")
set(BMOE_HAVE_LLAMA ON)
add_subdirectory("${BMOE_DIR}/core" "${CMAKE_CURRENT_BINARY_DIR}/bigmoeonedge")
target_compile_options(bmoe_core PRIVATE -march=armv8-a)

add_library(llmhub_llama_stream SHARED llama_stream_jni.cpp)
target_compile_options(llmhub_llama_stream PRIVATE -march=armv8-a -fvisibility=hidden)
target_link_options(llmhub_llama_stream PRIVATE "-Wl,--exclude-libs,ALL" "-Wl,-Bsymbolic")
target_link_libraries(llmhub_llama_stream PRIVATE bmoe_core llama llama-common mtmd ggml android log dl)

# The same engine and JNI against the official Snapdragon release libraries (above), so the prompt of a streamed MoE
# model can be read on the Hexagon NPU (LlamaCppStreamSnapdragonNative). libllama-common.so is built from this tree
# with the release flags by tools/llama_opencl/build.sh, like libllama.so.
foreach(prebuilt IN ITEMS ggml-base llama-common)
    add_library(llmhub_prebuilt_${prebuilt} SHARED IMPORTED)
    set_target_properties(llmhub_prebuilt_${prebuilt} PROPERTIES
        IMPORTED_LOCATION "${LLAMA_PREBUILT_LIB_DIR}/lib${prebuilt}.so")
endforeach()
get_target_property(BMOE_CORE_SOURCES bmoe_core SOURCES)
list(TRANSFORM BMOE_CORE_SOURCES PREPEND "${BMOE_DIR}/core/")
add_library(bmoe_core_snapdragon STATIC ${BMOE_CORE_SOURCES})
target_include_directories(bmoe_core_snapdragon PUBLIC "${BMOE_DIR}/core/include")
target_include_directories(bmoe_core_snapdragon PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/llama-prebuilt/include"
    "${LLAMA_CPP_DIR}/common"
    "${LLAMA_CPP_DIR}/vendor"
    "${LLAMA_CPP_DIR}/tools/mtmd")
target_compile_definitions(bmoe_core_snapdragon PUBLIC BMOE_HAVE_LLAMA=1 BMOE_VERSION="${PROJECT_VERSION}")
target_compile_definitions(bmoe_core_snapdragon PRIVATE __ANDROID_UNAVAILABLE_SYMBOLS_ARE_WEAK__)
target_compile_options(bmoe_core_snapdragon PRIVATE -march=armv8-a)
target_link_libraries(bmoe_core_snapdragon PUBLIC
    llmhub_prebuilt_llama llmhub_prebuilt_llama-common llmhub_prebuilt_mtmd llmhub_prebuilt_ggml
    llmhub_prebuilt_ggml-base android log dl)

add_library(llmhub_llama_stream_snapdragon SHARED llama_stream_jni.cpp)
target_compile_definitions(llmhub_llama_stream_snapdragon PRIVATE LLMHUB_SNAPDRAGON=1)
target_compile_options(llmhub_llama_stream_snapdragon PRIVATE -march=armv8-a -fvisibility=hidden)
target_include_directories(llmhub_llama_stream_snapdragon PRIVATE "${CMAKE_CURRENT_LIST_DIR}/llama-prebuilt/include")
target_link_libraries(llmhub_llama_stream_snapdragon PRIVATE bmoe_core_snapdragon)
target_link_options(llmhub_llama_stream_snapdragon PRIVATE "-Wl,--allow-shlib-undefined")

add_subdirectory(pocket_tts)
