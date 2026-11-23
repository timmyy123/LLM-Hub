#include "llmhub_bmoe.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <memory>
#include <string>

#include "bmoe/chat.h"
#include "bmoe/session.h"

struct llmhub_bmoe {
    std::unique_ptr<bmoe::Session> session;
    std::atomic<bool> stop{false};
    int last_prompt_tokens = 0;          // the last generate's prompt, as the engine read it
    double last_prompt_seconds = 0.0;
};

namespace {
void set_error(char * error, size_t size, const std::string & message) {
    if (error == nullptr || size == 0) return;
    std::snprintf(error, size, "%s", message.c_str());
}
} // namespace

extern "C" llmhub_bmoe * llmhub_bmoe_open(const char * model_path, const char * mmproj_path, int context_size,
                                          int threads, int gpu_prefill, int gpu_layers, char * error, size_t error_size) {
    bmoe::SessionConfig cfg;
    cfg.model_path = model_path != nullptr ? model_path : "";
    cfg.mmproj = mmproj_path != nullptr ? mmproj_path : "";
    cfg.n_ctx = std::max(512, context_size);
    // A streamed decode mostly waits on storage, so a few threads decode faster than many (on the Galaxy Tab S9+ 2
    // threads took 180 ms of compute per token and 6 threads 669 ms; iPhones have 2 performance cores). Prompt
    // reading is compute-bound and keeps the app's usual count.
    cfg.n_threads = std::min(2, std::max(1, threads));
    cfg.n_threads_batch = std::max(1, threads);
    cfg.n_batch = 512;
    cfg.n_ubatch = 512;
    cfg.chatml = false;                 // the app formats the chat and passes the finished prompt
    cfg.n_gpu_layers = 0;
    cfg.device_use = bmoe::DeviceUse::CpuOnly;   // unless gpu_layers > 0 (below)
    cfg.moe.enabled = true;
    cfg.moe.cache_auto = true;          // size the expert cache from the free memory
    // Anonymous memory for the always-resident weights: iOS has no reclaim-exempt memory for Pinned.
    cfg.moe.dense_weights = bmoe::DenseWeightsMode::Anonymous;
    cfg.moe.io_threads = 4;
    cfg.moe.o_direct = true;
    if (gpu_layers > 0) {
        // Layers on the GPU by the app's GPU layers setting. Only their dense weights go there: the expert weights stay
        // in host memory, where the streamer serves them (llama.cpp's -ot exps=CPU).
        cfg.n_gpu_layers = gpu_layers;
        cfg.device_use = bmoe::DeviceUse::Auto;
        cfg.buft_overrides = {"\\.ffn_(up|down|gate_up|gate)_(ch|)exps"};
    } else if (gpu_prefill != 0) {
        // Prompt reading on Metal, decode on the CPU: a one-token decode is faster on the cores, while a wide prefill
        // graph runs several times faster on the GPU (iPhone 15 Pro Max, LFM2.5-24B-A2B Q2_K_L, ~900 tokens: first
        // word 7.2 s against 57.8 s). The GPU's expert slots are held only while a prefill runs. No load-time check
        // (verified on devices in testing); a GPU that cannot open fails the load instead of silently using the CPU.
        cfg.n_batch = 2048;
        cfg.n_ubatch = 2048;
        cfg.prefill.device = "MTL0";
        cfg.prefill.slots_on_demand = true;
        cfg.prefill.verify = false;
        cfg.prefill.best_effort = false;
    }
    std::string message;
    auto session = bmoe::Session::open(cfg, message);
    if (!session) {
        set_error(error, error_size, message.empty() ? "flash streaming could not open the model" : message);
        return nullptr;
    }
    auto * handle = new llmhub_bmoe();
    handle->session = std::move(session);
    return handle;
}

extern "C" void llmhub_bmoe_free(llmhub_bmoe * session) {
    delete session;
}

extern "C" int llmhub_bmoe_vision(const llmhub_bmoe * session) {
    return session != nullptr && session->session->vision() ? 1 : 0;
}

extern "C" const char * llmhub_bmoe_image_marker(void) {
    return bmoe::image_marker();
}

extern "C" int llmhub_bmoe_generate(llmhub_bmoe * session, const char * prompt,
                                    const uint8_t * image, size_t image_size, const char * image_mime,
                                    int max_tokens, float temperature, int top_k, float top_p,
                                    llmhub_bmoe_piece_fn on_piece, void * user,
                                    int * generated_tokens, double * generate_seconds,
                                    char * error, size_t error_size) {
    if (generated_tokens != nullptr) *generated_tokens = 0;
    if (generate_seconds != nullptr) *generate_seconds = 0.0;
    if (session == nullptr || prompt == nullptr || *prompt == '\0') {
        set_error(error, error_size, "no model or empty prompt");
        return 1;
    }
    bmoe::GenerateRequest req;
    req.prompt = prompt;
    req.n_predict = std::max(1, max_tokens);
    req.clear_kv = true;                // each completion carries the whole conversation
    req.render_text = false;
    session->stop.store(false);
    req.stop = &session->stop;
    bmoe::RequestSampling sampling;
    sampling.temp = std::max(0.0f, temperature);
    sampling.top_k = std::max(1, top_k);
    sampling.top_p = std::clamp(top_p, 0.0f, 1.0f);
    req.sampling = sampling;
    if (image != nullptr && image_size > 0) {
        if (!session->session->vision()) {
            set_error(error, error_size, "this model has no loaded vision projector");
            return 2;
        }
        bmoe::ChatImage chat_image;
        chat_image.bytes.assign(image, image + image_size);
        chat_image.mime = image_mime != nullptr ? image_mime : "image/jpeg";
        req.images.push_back(std::move(chat_image));
    }
    const bmoe::RunResult result = session->session->generate(req, [&](const bmoe::TokenMetrics & metrics) {
        if (metrics.piece.empty() || on_piece == nullptr) return;
        if (on_piece(metrics.piece.data(), metrics.piece.size(), user) != 0) session->stop.store(true);
    });
    if (generated_tokens != nullptr) *generated_tokens = result.summary.n_generated;
    if (generate_seconds != nullptr) *generate_seconds = result.summary.gen_seconds;
    session->last_prompt_tokens = result.summary.n_prompt;
    session->last_prompt_seconds = result.summary.prefill_seconds;
    if (!result.ok) {
        set_error(error, error_size, result.error.empty() ? "flash streaming generation failed" : result.error);
        return 3;
    }
    return 0;
}

extern "C" void llmhub_bmoe_cancel(llmhub_bmoe * session) {
    if (session == nullptr) return;
    session->stop.store(true);
    session->session->cancel();
}

extern "C" void llmhub_bmoe_last_prompt(const llmhub_bmoe * session, int * tokens, double * seconds) {
    if (tokens != nullptr) *tokens = session != nullptr ? session->last_prompt_tokens : 0;
    if (seconds != nullptr) *seconds = session != nullptr ? session->last_prompt_seconds : 0.0;
}
