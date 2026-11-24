// A persistent engine session: load the model and warm the expert cache ONCE, then serve
// many prompts against that resident state.
//
// run() (runtime.h) is the one-shot convenience — it opens a Session, generates once, and
// closes. Session exists for the interactive case: a fresh process per prompt re-pays the
// model load (tens of seconds) and the whole expert-cache warm-up ramp every time, which is
// exactly what streaming was meant to avoid. Keeping the process alive amortises both:
//
//   auto s = Session::open(cfg, err);          // model load + capture + source.init, once
//   s->generate({.prompt = "..."});            // prefill + decode; cache survives
//   s->generate({.prompt = "..."});            // starts warm — no reload, no cold ramp
//
// This header is pure policy (no llama.cpp types) — the native state lives behind a pimpl,
// so the CLI and tests link it without pulling in the backend headers.
#pragma once

#include "bmoe/chat.h"
#include "bmoe/config.h"
#include "bmoe/decide.h"
#include "bmoe/metrics.h"
#include "bmoe/runtime.h"

#include <atomic>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace bmoe {

class IRouteTraceSink;
class IComputeTraceSink;
class IIoTraceSink;

// Everything fixed for the model's lifetime — set once at open(). n_ctx and n_batch are
// baked into the llama context at creation and cannot change per prompt, so size them for
// the longest prompt+generation the session will serve.
struct SessionConfig {
    std::string model_path;
    // The multimodal projector that gives this model its image input (an `mmproj` GGUF from the same
    // conversion), loaded beside the model. Empty: text only, and a request with images is refused.
    // See RunConfig::mmproj.
    std::string mmproj;
    int image_max_tokens = 0; // the most tokens one image may take, 0 = the projector's own cap. See RunConfig.
    int n_threads = 4;
    int n_threads_batch = 0; // 0 = same as n_threads. See RunConfig::n_threads_batch.
    int n_ctx = 2048;
    int n_batch = 512; // prefill chunk capacity; longer prompts are prefilled in n_batch slices
    // Widest graph actually computed at once. 0 = follow n_batch. Sizing this down trades prefill
    // throughput for resident compute buffers, which on this engine compete with the expert cache.
    // See RunConfig::n_ubatch.
    int n_ubatch = 0;
    int ctx_checkpoints = 8;                            // See RunConfig::ctx_checkpoints.
    int checkpoint_min_step = 8192;                     // See RunConfig::checkpoint_min_step.
    KvCacheType cache_type_k = k_default_kv_cache_type; // See RunConfig::cache_type_k.
    KvCacheType cache_type_v = k_default_kv_cache_type;
    bool chatml = false;
    // Active-expert (top-k) override applied at load via a kv_override on the arch-prefixed
    // expert_used_count key. 0 = use the model's own count. See RunConfig::n_expert_used.
    int n_expert_used = 0;
    // The first stage's placement, applied at load. See RunConfig::n_gpu_layers.
    int n_gpu_layers = 0;
    std::vector<std::string> buft_overrides;
    std::vector<std::pair<std::string, int>> device_split; // See RunConfig::device_split.
    std::vector<std::string> device_list;                  // See RunConfig::device_list.
    // See RunConfig::device_use.
    DeviceUse device_use = DeviceUse::Auto;
    SamplingConfig sampling; // fixed for the session; greedy by default (temp <= 0)
    MoeStreamConfig moe;
    // Self-speculative decoding, and which source drafts. Fixed for the session: it decides whether
    // open() builds the wider verify batch, and — for the MTP source only — the draft context.
    // See RunConfig::spec.
    SpecConfig spec;
    // Wide prefill graphs on an accelerator, decode on the CPU. Fixed for the session: the device
    // joins the scheduler at load. See RunConfig::prefill.
    PrefillDeviceConfig prefill;
    DecideConfig decide;   // decide()'s prefix-state policy. See RunConfig::decide.
    KnobTrialConfig trial; // lossless knobs measured on the first decodes. See RunConfig::trial.
    // What the session opens where the device this one names does not keep the model. See
    // RunConfig::fallback; empty, the same configuration on the cores.
    std::shared_ptr<const SessionConfig> fallback;
    bool llama_debug_log = false; // See RunConfig::llama_debug_log.
};

// The RunConfig → SessionConfig mapping, in one place. Both entry points that open a session from a
// RunConfig — run() and the CLI's interactive loop — need it, and they used to spell it out field by
// field. Two copies of a mapping is one copy too many: adding a field to RunConfig must not depend on
// remembering to touch both. n_batch = n_ctx so any prompt that fits the context prefills in one batch.
SessionConfig session_config_from(const RunConfig & cfg);

// How a GenerateRequest::think=false request can be honoured on THIS model. Decided once at
// open() by rendering the model's own chat template, never from a list of model names.
//
// Turning thinking off is a request to the template, and a template is free to ignore it: the
// flag reaches the jinja context either way, so a model whose template never reads it reasons on
// regardless and nothing reports that the setting was dropped. Probing says which case a model is
// in, so a caller can stop offering a control that does nothing.
enum class ThinkControl {
    // Passing the flag is all there is to do — either the template reads it, or the model does not
    // reason at all and there is nothing to suppress. Both leave the engine with nothing to add.
    Template,
    Prefill, // the template ignores it; the turn starts past the reasoning section instead
    None,    // the model reasons on every turn and cannot be asked not to
};

// Stable lowercase name ("template", "prefill", "none") for logs and the telemetry protocol.
const char * think_control_name(ThinkControl c);

// Where a turn's prefill stands (GenerateRequest::on_prefill). `total` is the whole prompt in
// tokens, `cached` the part the context already held (reused, not decoded again), `processed` the
// part decoded so far this turn; `seconds` since the prefill began.
struct PrefillProgress {
    int total = 0;
    int cached = 0;
    int processed = 0;
    double seconds = 0.0;
};

// One request's own sampling, over the session's (SamplingConfig): a field left unset keeps the
// session's value. A request that sets nothing here samples as the session does. temp <= 0 is greedy.
// The penalties and the bias are applied before the cutoffs, as llama.cpp's sampler chain applies
// them; a bias of -INFINITY forbids a token. With speculation armed (SpecConfig) the session decodes
// greedily, since a draft is accepted by the target's argmax, and a request's sampling is not applied.
struct RequestSampling {
    std::optional<float> temp, top_p, min_p;
    std::optional<int> top_k;
    std::optional<uint32_t> seed;
    float repeat_penalty = 1.0f;    // 1: off
    float frequency_penalty = 0.0f; // 0: off
    float presence_penalty = 0.0f;  // 0: off
    int penalty_last_n = 64;        // how many recent tokens the penalties look back over (-1: the context)
    std::vector<std::pair<int32_t, float>> logit_bias;
};

// Per-prompt request. clear_kv=true (the default) makes each prompt independent while the
// expert cache stays warm; clear_kv=false continues the KV cache for multi-turn chat.
struct GenerateRequest {
    std::string prompt;
    int n_predict = 32;
    bool think = false; // RunConfig::think's default
    bool clear_kv = true;
    // Populate TokenMetrics::text / ::reasoning on every token. Building them means parsing the
    // WHOLE generation so far — the chat parser cannot resume — so it is O(n) per token and O(n²)
    // over a turn, and in chat mode it also allocates a copy of everything generated. A UI that
    // renders a running answer needs it; a consumer that only concatenates `piece` (the CLI's
    // default output, and every benchmark run) does not, and should turn it off. Default on so an
    // embedder that does not know about this flag keeps the old behaviour.
    bool render_text = true;
    // Serialized JSON Schema for this turn; empty leaves the template unconstrained.
    // The template builds both the sampling grammar and the response parser from it.
    std::string json_schema{};

    // The whole conversation, for a front-end that keeps its own record of it (a chat protocol
    // whose client resends the history on every request, an edited or regenerated turn). When
    // non-empty it REPLACES the engine-held conversation and `prompt` is not read: the last message
    // is the one being answered. The KV is not dropped for it - with clear_kv=false the tokens
    // still matching what the context holds are kept and only the rest is prefilled, so resending
    // the conversation the session already has costs the new turn alone, and a different one
    // prefills from its first differing token. Chat mode only.
    std::vector<ChatMessage> messages{};
    // Functions the model may call this turn, described to it by its own chat template; the calls
    // come back in RunResult::tool_calls. Read only with `messages`: a tool's answer is a message.
    std::vector<ChatTool> tools{};
    ToolChoice tool_choice = ToolChoice::Auto;
    // The images `prompt` places with image_marker(), in order (the raw-prompt path; a message list
    // carries each message's own). Needs a session opened with a projector. Kept after the earlier
    // fields so existing positional aggregate initializers still read the same.
    std::vector<ChatImage> images{};
    // Optional caller-owned stop flag. It must outlive generate(); the caller may raise it from
    // another thread. Unlike Session::cancel()'s latch, it is never reset by the session.
    const std::atomic<bool> * stop = nullptr;
    // Told where the prompt's prefill stands: once before the first piece (processed 0) and after
    // each piece decoded, on the generating thread. Nullable. A front-end shows it while a long
    // prompt goes in, the stretch of a turn with no token to show.
    std::function<void(const PrefillProgress &)> on_prefill{};
    // This request's sampling over the session's (RequestSampling); unset samples as the session does.
    std::optional<RequestSampling> sampling{};
    // Text that ends the answer where it first appears; the answer stops before it (`stop` above is
    // the caller's cancel flag, not this). Matched in the answer only: reasoning written before it is
    // not searched, or a stop the reasoning contains would end the turn with no answer. While the text
    // generated so far ends partway into one, that part is held back from TokenMetrics::text, so a
    // streamed answer never shows a stop it then takes back.
    std::vector<std::string> stop_text{};
    // A GBNF grammar the whole answer must follow (rule `root`), in llama.cpp's grammar language.
    // Exclusive of `tools` and `json_schema`, which build their own. The model is asked not to
    // reason first; a token the grammar cannot take fails the turn, as for a response schema.
    std::string grammar{};
    // Report the top n_probs candidates and the chosen token's log-probability on every token
    // (TokenMetrics::top_logprobs). 0: none.
    int n_probs = 0;
};

// Teacher-forced quality measurement. The text is fixed, so every cell scores the SAME token
// sequence and no cell can fork onto its own trajectory — which is what makes the number a scale
// rather than a coin flip. Comparing generated text instead does not work: measured on this
// engine, capping one layer's routing to 1 expert of 8 left the greedy output byte-identical while
// capping it to 7 changed it, because greedy only moves when a perturbation happens to cross an
// argmax boundary, and whether it does is unrelated to how large the perturbation was.
struct PplResult {
    bool ok = false;
    std::string error;
    double nll = 0.0; // mean negative log-likelihood per scored token, in nats
    double ppl = 0.0; // exp(nll)
    int n_scored = 0; // tokens that contributed (the first `skip` and the last are not scored)
    // Scored positions where the model's most likely token WAS the one that follows. Perplexity is
    // the sensitive metric and this is the legible one: it says how often the model would have
    // written the text itself, which a reader can picture without knowing what a nat is.
    int n_top1 = 0;
    int n_tokens = 0; // tokens in the text
    // Log-probability of each PplRequest::choices entry (its first token) at the position after
    // the whole text, in the request's order. Empty unless choices were given. This is what a
    // multiple-choice benchmark compares: the text is the question with its options, and the
    // choices are the answer letters.
    std::vector<double> choice_logp;
    double seconds = 0.0;
    // What the routing policy actually did during the scoring pass. A lossy setting that reports
    // 0 touched slots was inert, and the perplexity it produced is the baseline's — which is
    // exactly how a measurement quietly says nothing.
    long long experts_routed = 0;
    long long experts_dropped = 0;
    long long experts_reranked = 0; // slots the substitution examined (its own denominator)
    long long experts_substituted = 0;
};

struct PplRequest {
    std::string text;
    // Tokens at the start of the text whose prediction is made from too little context to be
    // informative. They are evaluated (the KV needs them) but not scored.
    int skip = 8;
    // Score with the routing policies armed as they are during DECODE. A perplexity pass is one
    // wide batch, which the engine would otherwise label prefill — and the lossy routing policies
    // are decode-only by default, so the measurement would run with the very policy it is meant
    // to price switched off.
    bool as_decode = true;
    // Score one token per decode instead of the whole text in one batch. A wide batch routes
    // every position of a layer before any of it is read, so the cache holds whatever the previous
    // layers left and a policy that consults residency (dropping, substitution) barely fires: on a
    // 2000 MiB budget it touched 1-3% of the slots it examined, against 30% and more in
    // generation. Stepping reproduces the decode regime exactly — one token, a warm cache shaped
    // by the previous tokens of the same text — at the cost of one decode per token. The first
    // `skip` tokens are fed as one prefill batch, and are not scored either way.
    bool step = false;
    // Strings whose first token's log-probability is reported at the end of the text. The text is
    // fed in full (its last token too), so the reported distribution is the one a generation
    // would sample its first token from.
    std::vector<std::string> choices;
};

class Session {
public:
    ~Session();

    // Load the model, discover the MoE expert tensors, and initialise the expert source.
    // Returns nullptr and sets `error` on failure. The returned session owns all native
    // state and must outlive every generate() call.
    //
    // The trace sinks (all nullable) turn on their diagnostic for every generate() on this session
    // and must outlive it. None is ever on for a benchmark run — each perturbs what it measures.
    //   * route_trace   — per-step, per-layer routing; ignored when streaming is off (no routing to
    //                     observe). See bmoe/route_trace.h.
    //   * compute_trace — per-node compute and fault attribution. Works with or without streaming,
    //                     so a dense mmap baseline can be compared against a streamed run.
    //   * io_trace      — per-read flash latency/size; ignored when streaming is off (no reads).
    // See bmoe/decode_trace.h for the latter two.
    static std::unique_ptr<Session> open(const SessionConfig & cfg,
                                         std::string & error,
                                         IRouteTraceSink * route_trace = nullptr,
                                         IComputeTraceSink * compute_trace = nullptr,
                                         IIoTraceSink * io_trace = nullptr);

    // Generate one response. Serialized: one generation at a time per session. `on_token`
    // and `sink` receive the same per-token metrics as run(). Cache state carries over from
    // the previous call. If cancel() or req.stop ends the turn, returns ok=true with cancelled=true.
    RunResult generate(const GenerateRequest & req,
                       const std::function<void(const TokenMetrics &)> & on_token = nullptr,
                       IMetricsSink * sink = nullptr);

    // Request the in-flight generation to stop. Thread-safe; may be called from another
    // thread while generate() runs. Takes effect at an existing decode/encoder cancellation
    // point and leaves the model/cache intact for the next generate().
    void cancel();

    double load_seconds() const;      // model load + streaming setup, measured once at open()
    const std::string & arch() const; // model architecture ("qwen3moe", "gemma4", …)
    int n_ctx() const;

    // Experts this model actually routes per token, after any n_expert_used override — the width a
    // caller needs to interpret MoeStreamConfig::drop_cold_frac, whose threshold is a fraction of
    // 1/top-k. 0 when the model is not MoE or the count could not be read.
    int n_expert_used() const;

    // The routing this session recorded so far (MoeStreamConfig::routing_record_tokens), as a value
    // for the front-end to keep against the model's fingerprint and hand to the next plan
    // (AutoPlanOptions::routing): the hit curve the cache is priced on (bmoe/routing.h). Empty
    // without streaming or with the record off. Call it between generations.
    RoutingTrace routing_record() const;

    // What the knob trial (SessionConfig::trial) has settled so far, as a value for the front-end to
    // keep against the model's fingerprint, stamped with this machine's hardware_fingerprint and the
    // engine version (bmoe/measured_knobs.h), and hand to the next plan (AutoPlanOptions::knobs).
    // Each knob appears once its verdict is in; empty with no trial or before the first verdict.
    // Call it between generations.
    MeasuredKnobs measured_knobs() const;

    // Which thinking-off mechanism this model supports (probed at open()). Report it to the user
    // rather than leaving a Thinking toggle that silently does nothing. Always Template when chat
    // mode is off, where no template is rendered and the question does not arise.
    ThinkControl think_control() const;

    // Whether this session reads images: a projector was given at open() and it has a vision
    // encoder. What a front-end asks before offering an image control.
    bool vision() const;

    // The model's tokenizer, outside any turn. Read-only on the vocabulary, so safe beside a running
    // generate() on another thread, as llama.cpp's own server tokenizes beside its decode loop.
    // `parse_special` reads control-token text ("<|im_start|>") as the token it names.
    std::vector<int32_t> tokenize(const std::string & text, bool add_special, bool parse_special) const;
    // The text of one token, special tokens spelled out. "" for an id outside the vocabulary.
    std::string token_piece(int32_t token) const;
    // The text of `tokens`, joined as the model would print them. False for an id outside the
    // vocabulary, with `text` untouched.
    bool detokenize(const std::vector<int32_t> & tokens, std::string & text) const;
    // The prompt generate() would decode for req.messages (with its tools, tool_choice, think and
    // json_schema), rendered by the model's own template; the engine-held conversation is not read
    // or changed. False with `error` set when chat mode is off or the template refuses it. Safe
    // beside a running generate(), like tokenize().
    bool render_chat(const GenerateRequest & req, std::string & prompt, std::string & error) const;

    // Set the expert-cache budget in MiB and evict down to it now. PRECONDITION: no generate() in
    // flight — call it between generations (e.g. from an app's memory-pressure callback). A no-op
    // when the cache is off. The only way the budget moves after init sizes it.
    void set_cache_budget_mb(int mib);

    // Score a fixed text under teacher forcing. Clears the KV; leaves no conversation state behind.
    PplResult perplexity(const PplRequest & req);

    // Read one decision (bmoe/decide.h): which of req.choices the model would answer next, from a
    // single prefill and no decode. Ends any conversation generate() was continuing — a decision is
    // not a turn — and keeps only the prefix state, which the next decide() may restore.
    // cancel() stops it between prefill chunks. Refused (not fatal) unless the session was opened
    // with SessionConfig::decide.enabled.
    DecideResult decide(const DecideRequest & req);

private:
    Session();
    // One opening of a configuration as it stands; open() opens the cores after it where a device's
    // answers for the model were not confirmed (KnobTrialConfig::answer_device).
    struct AnswerCheck;
    static std::unique_ptr<Session> open_placed(const SessionConfig & cfg,
                                                std::string & error,
                                                IRouteTraceSink * route_trace,
                                                IComputeTraceSink * compute_trace,
                                                IIoTraceSink * io_trace,
                                                AnswerCheck * check);
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace bmoe
