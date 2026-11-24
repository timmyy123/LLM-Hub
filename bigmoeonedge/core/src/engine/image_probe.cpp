#include "image_probe.h"

#include <climits>

// Static and private to this translation unit: the projector library carries its own copy, and
// neither copy is a public symbol of either.
#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_STDIO
#define STBI_NO_LINEAR
#define STBI_NO_HDR
// A static copy declares every reader; the probe calls one of them.
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#endif
#include "stb/stb_image.h"
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

namespace bmoe::detail {

bool image_dimensions(const unsigned char * bytes, size_t n, int & width, int & height) {
    if (n == 0 || n > (size_t) INT_MAX) return false;
    int comp = 0;
    return stbi_info_from_memory(bytes, (int) n, &width, &height, &comp) != 0;
}

} // namespace bmoe::detail
