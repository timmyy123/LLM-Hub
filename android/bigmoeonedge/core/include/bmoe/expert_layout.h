// Which tensors of a model are its routed experts, read from the file instead of from a table.
//
// This is the only architecture-shaped knowledge the engine holds, and it is knowledge of no
// architecture: a routed expert weight is a tensor of a block, a `.weight`, three-dimensional, whose
// third extent is the expert count the file declares (`<arch>.expert_count`). That is how llama.cpp
// lays out every weight it hands to the expert matmul (`mul_mat_id`), whatever the model, and it is
// what told the architectures a registry used to name apart from the rest of their files: the shared
// expert is 2-D, the router is 2-D, a per-expert scale is 1-D, the router's bias is 1-D. The expert
// count, the per-expert stride and the file offsets were already read from the file; now the names
// are too, so a MoE family llama.cpp learns to load streams here with no change, and a file that lays
// its experts out in a way this engine has not proved byte-identical is refused with the reason
// rather than guessed at.
//
// What a layout is: the expert tensors of each MoE layer, in slots. Every MoE layer of a file names
// the same set of suffixes (gate, up and down; a fused gate_up with down; up with down), and a slot
// is a suffix's position in that set, sorted, so the same slot holds the same tensor in every layer.
// The engine streams whatever a slot holds and never needs to know which projection it is. Layers
// with no expert tensor - leading dense blocks, the attention or recurrent blocks of a hybrid stack -
// are not MoE layers and never bind.
//
// Pure policy: no llama.cpp, no ggml. The facts arrive as a list of (name, third extent) the gguf
// reader fills from the file.
#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace bmoe {

// One tensor as the file states it, reduced to the two facts the discovery reads.
struct TensorExtent {
    std::string name;
    uint64_t third = 0; // ne[2] of a tensor with exactly three dimensions; 0 for any other rank
};

struct ExpertLayout {
    // The most expert tensors a layer may name. Three is every layout llama.cpp builds: the split
    // {gate, up, down}, the fused {gate_up, down}, the gate-less {up, down}. A file with more is
    // refused, not truncated: the streamer's per-slot buffers are sized by this.
    static constexpr int max_slots = 3;

    bool ok = false; // a layout this engine streams; when false, `error` says why not
    std::string error;
    int n_expert = 0;              // the file's routed expert count
    int n_slots = 0;               // expert tensors per MoE layer, 1..max_slots
    std::string suffix[max_slots]; // the slot order every MoE layer shares: "blk.<il>.<suffix>.weight"
    std::vector<int> layers;       // the MoE layers, ascending

    // The slot expert tensor `name` fills and its layer, or -1 when `name` is not one of this layout's
    // expert tensors. Exactly "blk.<il>.<suffix>.weight": the ".weight" tail is load-bearing, since a
    // companion like "ffn_down_exps.scale" shares the suffix and must stay a resident, mapped tensor.
    // No allocation: it runs on every graph node of the capture pass.
    int slot_of(const char * name, int & il_out) const;
};

// Read the routed experts off the file's tensor list. `n_expert` is the count the file declares; a
// file declaring none (a dense model) or one is refused as not a MoE, and so is one whose MoE layers
// do not all name the same expert tensors, or name more than max_slots. The refusal carries its
// reason, phrased for the person who passed the file.
ExpertLayout discover_expert_layout(const std::vector<TensorExtent> & tensors, int n_expert);

// What one expert of each layer costs a cache, in bytes, from the file alone: each expert tensor's
// size over its expert count (its nb[2]), summed over the layer's slots. Indexed by layer, `n_layer`
// long, 0 for a layer with no routed experts. Where the plan put a layer's experts does not change
// it, so a record of the routing sizes every MoE layer the same way whichever of them streamed.
// `size_by_name` is each tensor's data size, as the gguf reader states it.
std::vector<uint64_t> expert_bytes_per_layer(const ExpertLayout & layout,
                                             const std::vector<TensorExtent> & tensors,
                                             const std::unordered_map<std::string, uint64_t> & size_by_name,
                                             int n_layer);

} // namespace bmoe
