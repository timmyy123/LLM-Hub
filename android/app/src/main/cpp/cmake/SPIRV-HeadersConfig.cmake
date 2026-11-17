# llama.cpp only requires this package's headers, and the Android NDK ships them. ggml-vulkan links
# the SPIRV-Headers::SPIRV-Headers target, so define it over the NDK's copy.
set(SPIRV-Headers_FOUND TRUE)
if(NOT TARGET SPIRV-Headers::SPIRV-Headers)
    add_library(SPIRV-Headers::SPIRV-Headers INTERFACE IMPORTED)
    set_target_properties(SPIRV-Headers::SPIRV-Headers PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
        "${CMAKE_ANDROID_NDK}/sources/third_party/shaderc/third_party/spirv-tools/external/spirv-headers/include")
endif()
