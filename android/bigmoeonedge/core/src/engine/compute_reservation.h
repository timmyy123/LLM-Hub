// The one way the engine asks llama.cpp to size a context's compute buffers again.
//
// A context reserves its compute memory when it is created, for the widest graph it may run, with
// every weight where the load put it. A session that later changes where wide graphs run (the
// prefill device rebinds the layer weights between graphs) keeps that first reservation for good
// unless the context redoes it, and llama.cpp's public API has no call that asks for that directly:
// it builds a new scheduler, and with it new buffers, when a setter changes something it treats as
// the graph's topology. So this asks through such a setter, changed and put back before the decode
// that reserves.
//
// Which setters do that is llama.cpp's behaviour, not its contract. `llama_set_causal_attn` did until
// upstream #28751, which ruled a causal flip no topology change; nothing failed, and every session
// with a prefill device kept the reservation made with every weight on the CPU, for a graph as wide
// as the ubatch that the CPU never runs. The gate `compute_reservation` checks the setter used here
// on every build, so a llama.cpp bump that changes it fails there instead of costing memory quietly.
#pragma once

#include "llama.h"

namespace bmoe::detail {

// The next llama_decode on `ctx` reserves its compute buffers afresh, for the weights as they are
// bound at that moment. `ctx` must compute no embeddings, as every context the session creates
// does: that is the value the request leaves in place.
void request_compute_reservation(llama_context * ctx);

} // namespace bmoe::detail
