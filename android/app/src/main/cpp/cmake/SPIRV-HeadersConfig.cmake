# llama.cpp only requires this package's headers; Android NDK includes them.
# CMakeLists.txt adds the NDK include path directly to ggml-vulkan.
set(SPIRV-Headers_FOUND TRUE)
