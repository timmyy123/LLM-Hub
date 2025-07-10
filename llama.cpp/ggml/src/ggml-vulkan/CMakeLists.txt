cmake_minimum_required(VERSION 3.19)
cmake_policy(SET CMP0114 NEW)

# The variable ${ggml_SOURCE_DIR} is set by the parent ggml/CMakeLists.txt
# and provides the most robust path to the common cmake file.
include(${ggml_SOURCE_DIR}/cmake/common.cmake)

if (GGML_VULKAN)
    message(STATUS "Including Vulkan backend")

    # ---------------------------------------------------------------------
    # Shader generation / embedding
    # ---------------------------------------------------------------------
    # By default we build the shader generator helper and run it at configure
    # time to convert the GLSL sources under vulkan-shaders/ into a pair of
    # C++ files (ggml-vulkan-shaders.hpp / .cpp) that get compiled into the
    # backend library.  This avoids having to distribute pre-compiled SPIR-V
    # blobs and works transparently on the Android NDK toolchain because the
    # NDK ships a `glslc` binary.

    # Detect Android cross-compilation 
    # When cross-compiling for Android, the shader generator tool gets built for the
    # target platform (Android ARM64) but needs to run on the host (Windows/Linux/Mac)
    # This causes the notorious "vulkan-shaders-gen not recognized" error
    set(IS_ANDROID_BUILD FALSE)
    if(CMAKE_SYSTEM_NAME STREQUAL "Android" OR 
       CMAKE_ANDROID_ARCH_ABI OR 
       ANDROID OR 
       CMAKE_TOOLCHAIN_FILE MATCHES "android")
        set(IS_ANDROID_BUILD TRUE)
    endif()

    if(IS_ANDROID_BUILD)
        # For Android builds, disable shader generation to avoid cross-compilation issues
        # See: https://github.com/ggerganov/llama.cpp/issues/11327
        message(WARNING "Vulkan cross-compilation has known issues. Disabling shader generation for Android builds.")
        message(WARNING "Consider using CPU backend for more reliable Android performance.")
        set(GGML_VULKAN_SHADER_GEN OFF)
    else()
        option(GGML_VULKAN_SHADER_GEN "Generate and embed Vulkan shaders" ON)
    endif()

    # The GLSL compiler.  If FindVulkan succeeded the variable will be set by
    # the top-level CMakeLists.txt.  Otherwise fall back to `glslc` in PATH.
    if (NOT Vulkan_GLSLC_EXECUTABLE)
        find_program(Vulkan_GLSLC_EXECUTABLE glslc)
    endif()

    set(SHADER_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}")
    set(SHADER_HPP "${SHADER_OUTPUT_DIR}/ggml-vulkan-shaders.hpp")
    set(SHADER_CPP "${SHADER_OUTPUT_DIR}/ggml-vulkan-shaders.cpp")

    if (GGML_VULKAN_SHADER_GEN)
        add_subdirectory(vulkan-shaders)

        set(SHADER_GEN_EXE vulkan-shaders-gen)
        set(SHADER_INPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/vulkan-shaders")

        add_custom_command(
            OUTPUT ${SHADER_HPP} ${SHADER_CPP}
            COMMAND $<TARGET_FILE:${SHADER_GEN_EXE}>
                    --input-dir ${SHADER_INPUT_DIR}
                    --output-dir ${SHADER_OUTPUT_DIR}
                    --target-hpp ${SHADER_HPP}
                    --target-cpp ${SHADER_CPP}
                    --glslc ${Vulkan_GLSLC_EXECUTABLE}
            DEPENDS ${SHADER_GEN_EXE}
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/vulkan-shaders
            COMMENT "Generating Vulkan shader header and source via ${SHADER_GEN_EXE}"
            VERBATIM
        )

        add_custom_target(ggml-vulkan-shaders_gen DEPENDS ${SHADER_HPP} ${SHADER_CPP})
    else()
        # Create empty stub files for Android builds to satisfy compilation
        # This disables Vulkan functionality but allows the build to complete
        file(WRITE ${SHADER_HPP} "// Empty Vulkan shader stubs for Android cross-compilation\n// See: https://github.com/ggerganov/llama.cpp/issues/11327\n#pragma once\n")
        file(WRITE ${SHADER_CPP} "// Empty Vulkan shader stubs for Android cross-compilation\n// See: https://github.com/ggerganov/llama.cpp/issues/11327\n")
        
        # Create a dummy target to satisfy dependencies
        add_custom_target(ggml-vulkan-shaders_gen 
            COMMENT "Using empty Vulkan shader stubs for Android build"
        )
    endif()

    # ---------------------------------------------------------------------
    # Backend library
    # ---------------------------------------------------------------------
    set(GGML_VULKAN_SOURCES
        ggml-vulkan.cpp
    )

    # Add generated shader source if shader generation is enabled.
    if (GGML_VULKAN_SHADER_GEN)
        list(APPEND GGML_VULKAN_SOURCES ${SHADER_CPP})
    endif()

    ggml_add_backend_library(ggml-vulkan ${GGML_VULKAN_SOURCES})

    # Always add the dependency and include directory for consistency
    add_dependencies(ggml-vulkan ggml-vulkan-shaders_gen)
    target_include_directories(ggml-vulkan PRIVATE ${SHADER_OUTPUT_DIR})

    target_compile_definitions(ggml-vulkan PRIVATE GGML_VULKAN)

    if (MSVC)
        target_compile_options(ggml-vulkan PRIVATE /W4 /utf-8)
    else()
        target_compile_options(ggml-vulkan PRIVATE -Wall -Wextra -Wpedantic -Wcast-qual -Wdouble-promotion -Wshadow -Wstrict-prototypes -Wpointer-arith -Wmissing-prototypes -Werror=incompatible-pointer-types -Wno-unused-function)
    endif()
endif()
