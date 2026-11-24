#pragma once

// bmoe::KvCacheType to the ggml type llama.cpp's context takes for it. The names and the set are
// llama.cpp's own (common's --cache-type-k / --cache-type-v), so the mapping is one to one.

#include "bmoe/config.h"

#include "ggml.h"

namespace bmoe::detail {

inline ggml_type to_ggml_type(KvCacheType t) {
    switch (t) {
    case KvCacheType::F16:
        return GGML_TYPE_F16;
    case KvCacheType::BF16:
        return GGML_TYPE_BF16;
    case KvCacheType::F32:
        return GGML_TYPE_F32;
    case KvCacheType::Q8_0:
        return GGML_TYPE_Q8_0;
    case KvCacheType::Q5_1:
        return GGML_TYPE_Q5_1;
    case KvCacheType::Q5_0:
        return GGML_TYPE_Q5_0;
    case KvCacheType::Q4_1:
        return GGML_TYPE_Q4_1;
    case KvCacheType::Q4_0:
        return GGML_TYPE_Q4_0;
    case KvCacheType::IQ4_NL:
        return GGML_TYPE_IQ4_NL;
    }
    return GGML_TYPE_F16;
}

} // namespace bmoe::detail
