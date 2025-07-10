# detect_host_compiler.cmake
#
# This script detects a suitable C and C++ compiler on the host system,
# which is necessary when cross-compiling. It sets the following variables:
#
#   HOST_C_COMPILER:   path to the host C compiler
#   HOST_CXX_COMPILER: path to the host C++ compiler
#

function(detect_host_compiler HOST_C_COMPILER_VAR HOST_CXX_COMPILER_VAR)
    if (DEFINED ENV{HOST_CC} AND DEFINED ENV{HOST_CXX})
        set(${HOST_C_COMPILER_VAR} "$ENV{HOST_CC}" PARENT_SCOPE)
        set(${HOST_CXX_COMPILER_VAR} "$ENV{HOST_CXX}" PARENT_SCOPE)
        message(STATUS "Using host compilers from environment: CC=${HOST_C_COMPILER_VAR}, CXX=${HOST_CXX_COMPILER_VAR}")
        return()
    endif()

    if (CMAKE_HOST_SYSTEM_NAME STREQUAL "Windows")
        # When cross-compiling for Android from Windows, CMake might find the NDK's clang.exe first.
        # We need to explicitly find cl.exe. Since you are running from the VS dev prompt,
        # it should be in the PATH.
        find_program(C_COMPILER_PATH NAMES cl.exe)
        find_program(CXX_COMPILER_PATH NAMES cl.exe)
    elseif(CMAKE_HOST_UNIX)
        find_program(C_COMPILER_PATH NAMES gcc clang cc)
        find_program(CXX_COMPILER_PATH NAMES g++ clang++ c++)
    else()
        message(FATAL_ERROR "Unsupported host system for cross-compilation.")
    endif()

    if(NOT C_COMPILER_PATH OR NOT CXX_COMPILER_PATH)
        message(FATAL_ERROR "Could not find a native host compiler. "
                            "Please ensure it is in your PATH, or set HOST_CC and HOST_CXX environment variables.")
    endif()

    set(${HOST_C_COMPILER_VAR} "${C_COMPILER_PATH}" PARENT_SCOPE)
    set(${HOST_CXX_COMPILER_VAR} "${CXX_COMPILER_PATH}" PARENT_SCOPE)
endfunction() 