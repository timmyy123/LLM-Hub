#pragma once

#include "backend.h"
#include "prefix_cache.h"

#include "bmoe/decide.h"

namespace bmoe::detail {

// One decision, start to finish, over any backend: render, check the choices, split at the prefix,
// restore what the cache holds, prefill the rest, keep the new prefix state, score. `cache` may be
// null (no prefix state is kept, every prompt is prefilled whole).
//
// The backend's sequence is emptied on the way in, whatever happens next, and again on the way out:
// its memory then holds a whole prompt no later call continues from — decide() restores from the
// cache, and generate() expects either an empty sequence or its own conversation.
DecideResult run_decide(IDecideBackend & backend, IPrefixCache * cache, const DecideRequest & req);

} // namespace bmoe::detail
