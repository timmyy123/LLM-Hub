#include "compute_reservation.h"

namespace bmoe::detail {

void request_compute_reservation(llama_context * ctx) {
    // Embeddings change what the graph outputs, which llama.cpp answers with a new reservation on
    // every change of the flag. Two changes leave the flag as it was and one reservation pending.
    llama_set_embeddings(ctx, true);
    llama_set_embeddings(ctx, false);
}

} // namespace bmoe::detail
