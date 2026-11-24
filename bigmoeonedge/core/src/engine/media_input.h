// The engine's one adapter over llama.cpp's multimodal projector library (tools/mtmd): it opens a
// projector beside the model, turns a prompt's text and images into a PromptStream, and runs an
// image's encoder so its embeddings can be decoded into the context.
//
// Agnostic by construction: which image delimiters the model wants, how an image is tiled, how
// many tokens it takes and which positions they occupy (M-RoPE or not) are all read from the
// projector file by the library; nothing here names a model family or a format. The session owns
// the decode itself (placement, the arena, the route trace), so this type hands back batches and
// never calls llama_decode.
#pragma once

#include "bmoe/chat.h"
#include "prompt_stream.h"

#include "llama.h"

#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <vector>

struct mtmd_context;
struct mtmd_input_chunk;

namespace bmoe::detail {

class MediaInput {
public:
    // Open the projector `mmproj` for `model`. With `use_devices` the encoder and its weights go on
    // a device of the backends loaded, when one is registered (the library's choice, from the
    // registry); without, on the host. The caller decides by the rule its model follows.
    // `image_max_tokens` caps what one image may take where the projector's count follows the
    // image's size (0: the projector's own cap). `cancel`, when given, is read while an image is
    // encoded: raised, the encoder stops where it is and decode_media() fails, so a Stop pressed
    // during an image answers in a moment rather than when the encoder is done.
    static std::unique_ptr<MediaInput> open(const std::string & mmproj,
                                            const llama_model * model,
                                            int n_threads,
                                            bool use_devices,
                                            int image_max_tokens,
                                            const std::atomic<bool> * cancel,
                                            std::string & err);
    ~MediaInput();

    bool vision() const;

    // A serialized turn borrows its caller's flag; clear it before that caller can return.
    void set_stop(const std::atomic<bool> * stop) { watch_.stop = stop; }

    // The most pixels an image may decode to. Read from its header before any pixel is allocated:
    // a small file declaring 20000x20000 would otherwise decode to more than a gigabyte of RGB on
    // the way to the projector, which then downsizes it to its own few hundred pixels a side.
    // Generous on purpose, so no photo a phone or a camera takes is refused.
    static constexpr unsigned long long k_max_image_pixels = 64ull * 1024 * 1024;

    // The marker a prompt places an image with: the library's, the same string image_marker() gives.
    const char * marker() const;

    // A prompt tokenized with its images. The spans in `stream` are the chunks in `chunks`, in order.
    struct Tokenized {
        PromptStream stream;
        std::vector<mtmd_input_chunk *> chunks; // owned; freed with this object
        Tokenized() = default;
        Tokenized(const Tokenized &) = delete;
        Tokenized & operator=(const Tokenized &) = delete;
        ~Tokenized();
    };
    // Tokenize `text` with `images` placed at its markers. False with `err` when the request is
    // at fault (`invalid` set: the marker count and the image count differ, an image is not one the
    // projector reads, or one declares more than k_max_image_pixels), or the library failed on it.
    bool tokenize(const std::string & text,
                  const std::vector<ChatImage> & images,
                  bool add_special,
                  Tokenized & out,
                  std::string & err,
                  bool & invalid) const;

    // Why the images of `t` cannot be decoded into a context whose graph is `n_ubatch` tokens wide,
    // or empty when they can. An image whose tokens attend to each other both ways (the projector says
    // which, per image) has to go in one pass of the graph, and llama.cpp refuses a wider one; checked
    // before anything is encoded or decoded, so the request is refused with the session untouched.
    std::string width_refusal(const Tokenized & t, int n_ubatch) const;

    // Encode span k of `t` and hand its embeddings to `decode`, n_batch tokens at a time, positioned
    // from pos0 (the span's own 2-D positions under M-RoPE, pos0 + i otherwise). The model's
    // attention mode is switched for the span where its projector asks (some encoders are read with
    // non-causal attention) and restored after; such a span goes in one batch, and one wider than the
    // context's graph fails before it is encoded (width_refusal says why). Returns 0, or the first
    // non-zero `decode` result, or -1 when the encoder failed or the span does not fit the graph.
    int decode_media(llama_context * ctx,
                     const Tokenized & t,
                     size_t k,
                     llama_pos pos0,
                     int n_batch,
                     const std::function<int(llama_batch_ext *)> & decode) const;

private:
    MediaInput() = default;
    mtmd_context * ctx_ = nullptr;
    const llama_model * model_ = nullptr;

public:
    // What the encoder's eval callback reads: the session latch, the caller's flag, and how many nodes it has
    // been asked about, which paces its answers. Public for the callback alone.
    struct EncoderWatch {
        const std::atomic<bool> * cancel = nullptr;
        const std::atomic<bool> * stop = nullptr;
        bool cancelled() const {
            return (cancel && cancel->load(std::memory_order_relaxed)) ||
                   (stop && stop->load(std::memory_order_relaxed));
        }
        unsigned asked = 0;
    };

private:
    EncoderWatch watch_;
};

} // namespace bmoe::detail
