#include "media_input.h"
#include "image_probe.h"

#include "bmoe/log.h"
#include "bmoe/params.h"

#include "mtmd-helper.h"
#include "mtmd.h"

#include <algorithm>
#include <chrono>

namespace bmoe {

const char * image_marker() {
    return mtmd_default_marker();
}

} // namespace bmoe

namespace bmoe::detail {

namespace {

// The library logs through ggml's callback; routed to the engine's log so a service's log file
// gets it and a quiet front-end can drop it, as with llama.cpp's own lines.
void mtmd_to_bmoe_log(ggml_log_level level, const char * text, void *) {
    if (!text || !*text) return;
    std::string line(text);
    while (!line.empty() && (line.back() == '\n' || line.back() == '\r'))
        line.pop_back();
    if (line.empty()) return;
    const LogLevel lv = level == GGML_LOG_LEVEL_ERROR  ? LogLevel::Error
                        : level == GGML_LOG_LEVEL_WARN ? LogLevel::Warn
                                                       : LogLevel::Info;
    log_printf(lv, "projector: %s", line.c_str());
}

// The library's encoder has no abort hook; its scheduler's eval callback is the one way in, and the
// library passes it through. The scheduler asks, before it computes anything, which nodes of the
// graph are wanted, computes the nodes between two wanted ones in one piece, and after a wanted
// node asks once more, where a no ends the graph. A callback that wanted nothing would be heard
// once, at the start; wanting one node in every k makes the graph k-node pieces, each followed by
// the one question that can end it. k is the grain of a Stop: a few hundred nodes an image, so a
// few dozen pieces, and one scheduler sync each. An encode ended early returns with an output no
// one reads: decode_media() reads the latch before it decodes.
constexpr unsigned k_encoder_stop_grain = 16;

bool stop_encoder_on_cancel(ggml_tensor *, bool ask, void * user) {
    auto * w = static_cast<MediaInput::EncoderWatch *>(user);
    if (ask) return ++w->asked % k_encoder_stop_grain == 0;
    return !w->cancelled();
}

} // namespace

std::unique_ptr<MediaInput> MediaInput::open(const std::string & mmproj,
                                             const llama_model * model,
                                             int n_threads,
                                             bool use_devices,
                                             int image_max_tokens,
                                             const std::atomic<bool> * cancel,
                                             std::string & err) {
    mtmd_helper_log_set(mtmd_to_bmoe_log, nullptr);
    mtmd_context_params p = mtmd_context_params_default();
    // The library's own cap stands unless one is given: a fixed-count projector has no use for it.
    if (image_max_tokens > 0) p.image_max_tokens = image_max_tokens;
    std::unique_ptr<MediaInput> m(new MediaInput());
    m->watch_.cancel = cancel;
    if (cancel) {
        p.cb_eval = stop_encoder_on_cancel;
        p.cb_eval_user_data = &m->watch_; // the context is freed by ~MediaInput: the watch outlives it
    }
    // A device when wanted and the registry has one, the host otherwise: the library asks the
    // registry, so a GPU-less box and a laptop with a GPU run the same code with no branch here.
    p.use_gpu = use_devices;
    p.n_threads = n_threads;
    p.print_timings = false;
    p.warmup = false; // the first image pays the graph's allocation instead of every load paying it
    mtmd_context * c = mtmd_init_from_file(mmproj.c_str(), model, p);
    if (!c) {
        err = "the projector could not be loaded from " + mmproj +
              ": not a projector this model matches (the mmproj file is written by the same conversion as the "
              "model, for the same base model)";
        return nullptr;
    }
    m->ctx_ = c;
    m->model_ = model;
    return m;
}

MediaInput::~MediaInput() {
    if (ctx_) mtmd_free(ctx_);
}

bool MediaInput::vision() const {
    return ctx_ && mtmd_support_vision(ctx_);
}

const char * MediaInput::marker() const {
    return ctx_ ? mtmd_get_marker(ctx_) : mtmd_default_marker();
}

MediaInput::Tokenized::~Tokenized() {
    for (mtmd_input_chunk * c : chunks)
        mtmd_input_chunk_free(c);
}

bool MediaInput::tokenize(const std::string & text,
                          const std::vector<ChatImage> & images,
                          bool add_special,
                          Tokenized & out,
                          std::string & err,
                          bool & invalid) const {
    invalid = false;
    out.stream.clear();
    for (mtmd_input_chunk * c : out.chunks)
        mtmd_input_chunk_free(c);
    out.chunks.clear();

    std::vector<mtmd_bitmap *> bitmaps;
    bool ok = true;
    for (size_t i = 0; i < images.size() && ok; ++i) {
        const ChatImage & img = images[i];
        // The decoded size, from the header: refused before the decode, through the path an
        // unreadable image takes, so every API says why with the same status.
        int w_px = 0, h_px = 0;
        if (image_dimensions(reinterpret_cast<const unsigned char *>(img.bytes.data()), img.bytes.size(), w_px, h_px) &&
            (unsigned long long) w_px * (unsigned long long) h_px > k_max_image_pixels) {
            err = "image " + std::to_string(i + 1) + " is " + std::to_string(w_px) + "x" + std::to_string(h_px) +
                  " pixels, and this engine decodes at most " + std::to_string(k_max_image_pixels >> 20) +
                  " megapixels: send it downsized";
            invalid = true;
            ok = false;
            break;
        }
        mtmd_helper_bitmap_wrapper w = mtmd_helper_bitmap_init_from_buf(
            ctx_, reinterpret_cast<const unsigned char *>(img.bytes.data()), img.bytes.size(),
            /*placeholder*/ false, mtmd_helper_init_opt_default());
        if (w.video_ctx) mtmd_helper_video_free(w.video_ctx); // a still image needs none
        if (!w.bitmap) {
            err = "image " + std::to_string(i + 1) + " could not be decoded" +
                  (img.mime.empty() ? "" : " (" + img.mime + ")") + ": not an image format the projector reads";
            invalid = true;
            ok = false;
            break;
        }
        if (mtmd_bitmap_is_audio(w.bitmap)) {
            err = "input " + std::to_string(i + 1) + " is audio, and this engine reads images only";
            mtmd_bitmap_free(w.bitmap);
            invalid = true;
            ok = false;
            break;
        }
        bitmaps.push_back(w.bitmap);
    }

    // The count the library would refuse, checked here first: its reason is a log line, and the
    // request wants the mismatch as its own error.
    if (ok) {
        const std::string mk = marker();
        size_t n_markers = 0;
        for (size_t at = text.find(mk); at != std::string::npos; at = text.find(mk, at + mk.size()))
            ++n_markers;
        if (n_markers != images.size()) {
            invalid = true;
            err = "the prompt places " + std::to_string(n_markers) + " image marker(s) (" + mk + ") and carries " +
                  std::to_string(images.size()) + " image(s): one marker per image, in order";
            ok = false;
        }
    }
    mtmd_input_chunks * chunks = nullptr;
    if (ok) {
        mtmd_input_text in;
        in.text = text.c_str();
        in.text_len = text.size();
        in.add_special = add_special;
        in.parse_special = true;
        chunks = mtmd_input_chunks_init();
        const int32_t rc = mtmd_tokenize(ctx_, chunks, &in, bitmaps.data(), bitmaps.size());
        if (rc != 0) {
            err = "an image could not be prepared for the projector";
            ok = false;
        }
    }
    if (ok) {
        const size_t n = mtmd_input_chunks_size(chunks);
        for (size_t i = 0; i < n && ok; ++i) {
            const mtmd_input_chunk * c = mtmd_input_chunks_get(chunks, i);
            switch (mtmd_input_chunk_get_type(c)) {
            case MTMD_INPUT_CHUNK_TYPE_TEXT: {
                size_t nt = 0;
                const llama_token * toks = mtmd_input_chunk_get_tokens_text(c, &nt);
                out.stream.tokens.insert(out.stream.tokens.end(), toks, toks + nt);
                break;
            }
            case MTMD_INPUT_CHUNK_TYPE_IMAGE: {
                const char * id = mtmd_input_chunk_get_id(c);
                out.stream.push_media(mtmd_input_chunk_get_n_tokens(c), mtmd_input_chunk_get_n_pos(c), id ? id : "");
                out.chunks.push_back(mtmd_input_chunk_copy(c));
                break;
            }
            default:
                err = "the projector produced an input this engine does not decode (audio)";
                ok = false;
                break;
            }
        }
    }
    if (chunks) mtmd_input_chunks_free(chunks);
    for (mtmd_bitmap * b : bitmaps)
        mtmd_bitmap_free(b);
    if (!ok) {
        out.stream.clear();
        for (mtmd_input_chunk * c : out.chunks)
            mtmd_input_chunk_free(c);
        out.chunks.clear();
    }
    return ok;
}

namespace {

std::string one_pass_refusal(size_t image, int n_tokens, int n_ubatch) {
    return "image " + std::to_string(image + 1) + " takes " + std::to_string(n_tokens) +
           " tokens that attend to each other both ways, so it has to go in one pass of the graph, which is " +
           std::to_string(n_ubatch) + " tokens wide: set " + flag_spelling("ubatch") + " to at least " +
           std::to_string(n_tokens) + ", or " + flag_spelling("image-max-tokens") + " to at most " +
           std::to_string(n_ubatch);
}

} // namespace

std::string MediaInput::width_refusal(const Tokenized & t, int n_ubatch) const {
    for (size_t k = 0; k < t.chunks.size(); ++k) {
        const int n_tokens = (int) mtmd_input_chunk_get_n_tokens(t.chunks[k]);
        if (n_tokens > n_ubatch && mtmd_decode_use_non_causal(ctx_, t.chunks[k]))
            return one_pass_refusal(k, n_tokens, n_ubatch);
    }
    return std::string();
}

int MediaInput::decode_media(llama_context * ctx,
                             const Tokenized & t,
                             size_t k,
                             llama_pos pos0,
                             int n_batch,
                             const std::function<int(llama_batch_ext *)> & decode) const {
    if (k >= t.chunks.size() || n_batch <= 0) return -1;
    const mtmd_input_chunk * chunk = t.chunks[k];
    // Some encoders' tokens attend to each other both ways; the library says which, per chunk. Such
    // an image is one pass of the graph or nothing: checked before the encoder spends its time.
    const bool non_causal = mtmd_decode_use_non_causal(ctx_, chunk);
    const int n_ubatch = (int) llama_n_ubatch(ctx);
    if (non_causal && (int) mtmd_input_chunk_get_n_tokens(chunk) > n_ubatch) {
        log_printf(LogLevel::Error, "%s",
                   one_pass_refusal(k, (int) mtmd_input_chunk_get_n_tokens(chunk), n_ubatch).c_str());
        return -1;
    }
    const auto t0 = std::chrono::steady_clock::now();
    if (mtmd_encode_chunk(ctx_, chunk) != 0) return -1;
    // A cancel during the encode ended its graph early (stop_encoder_on_cancel): its output is
    // not an image, and the caller reads its own latch to tell a cancel from a failure.
    if (watch_.cancelled()) return -1;
    // What an image costs is mostly this: said once per image, so a slow answer has its reason.
    log_printf(LogLevel::Info, "image: %d tokens encoded in %.1f s", (int) mtmd_input_chunk_get_n_tokens(chunk),
               std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    const float * embd = mtmd_get_output_embd(ctx_);
    if (!embd) return -1;

    const int n_tokens = (int) mtmd_input_chunk_get_n_tokens(chunk);
    const size_t n_embd = (size_t) llama_model_n_embd_inp(model_);
    const bool mrope = mtmd_decode_use_mrope(ctx_);
    std::vector<mtmd_decoder_pos> pos2d;
    if (mrope) {
        const mtmd_image_tokens * it = mtmd_input_chunk_get_tokens_image(chunk);
        if (!it) return -1;
        pos2d.resize((size_t) n_tokens);
        mtmd_helper_image_get_decoder_pos(it, pos0, pos2d.data());
    }

    if (non_causal) llama_set_causal_attn(ctx, false);
    // Both ways means the whole image in one batch: split, each piece would attend to itself alone.
    const int step = non_causal ? n_tokens : n_batch;
    int rc = 0;
    llama_batch_ext * b = llama_batch_ext_init(ctx);
    for (int off = 0; off < n_tokens && rc == 0; off += step) {
        const int n = std::min(step, n_tokens - off);
        llama_batch_ext_clear(b);
        for (int i = off; i < off + n; ++i) {
            const llama_embd e = {embd + (size_t) i * n_embd, 1, n_embd};
            const int32_t idx = llama_batch_ext_add_embd(b, /*seq*/ 0, e);
            if (idx < 0) {
                rc = -1;
                break;
            }
            llama_pos p[4] = {0, 0, 0, 0};
            if (mrope) {
                p[0] = (llama_pos) pos2d[(size_t) i].t;
                p[1] = (llama_pos) pos2d[(size_t) i].y;
                p[2] = (llama_pos) pos2d[(size_t) i].x;
                p[3] = (llama_pos) pos2d[(size_t) i].z;
            } else {
                p[0] = pos0 + i;
            }
            llama_batch_ext_set_pos(b, idx, p);
            // The last image token keeps its logits: a prompt may end on an image.
            if (i == n_tokens - 1) llama_batch_ext_set_output_logits(b, idx, true);
        }
        if (rc == 0) rc = decode(b);
    }
    llama_batch_ext_free(b);
    if (non_causal) llama_set_causal_attn(ctx, true);
    return rc;
}

} // namespace bmoe::detail
