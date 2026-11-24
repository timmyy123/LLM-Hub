// The size an encoded image declares, read from its header alone: the check that refuses an image
// too large to decode runs before any pixel is allocated. A private, static copy of the stb_image
// reader the projector library decodes with (vendored under llama.cpp), so the probe reads exactly
// the formats the library will, and the library's own symbols stay its own.
#pragma once

#include <cstddef>

namespace bmoe::detail {

// True with the width and height when `bytes` has a header stb_image parses. False for a format it
// does not know (the library may still read it another way): a false here refuses nothing.
bool image_dimensions(const unsigned char * bytes, size_t n, int & width, int & height);

} // namespace bmoe::detail
