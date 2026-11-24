// Flash streaming runtime for MoE GGUF models (LlamaCppStreamNative in LlamaCppInferenceService.kt).
//
// The routed experts stay in the model file: BigMoeOnEdge's engine (android/bigmoeonedge, built on this app's
// own llama.cpp) reads the ones each token routes to from flash into an LRU cache sized from the free RAM, so a
// model larger than the phone's memory runs. Decode is on the CPU. The JNI surface matches the other llama.cpp
// runtimes: the app formats the chat, starts a completion with the formatted prompt, and pulls tokens one by one.
// Generation runs on a worker thread; nativeNextToken waits for its pieces.

#include <jni.h>
#include <android/log.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <fstream>
#include <iterator>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <unistd.h>

#include <cstdlib>

#include "bmoe/chat.h"
#include "bmoe/log.h"
#include "bmoe/session.h"
#include "ggml-backend.h"
#include "llama.h"

// Two builds of this file: libllmhub_llama_stream (the app's own llama.cpp, CPU) and, with LLMHUB_SNAPDRAGON,
// libllmhub_llama_stream_snapdragon (the official Snapdragon release libraries, so the prompt can be read on the
// Hexagon NPU).
#ifdef LLMHUB_SNAPDRAGON
#define STREAM_JNI(name) Java_com_llmhub_llmhub_inference_LlamaCppStreamSnapdragonNative_##name
#else
#define STREAM_JNI(name) Java_com_llmhub_llmhub_inference_LlamaCppStreamNative_##name
#endif

namespace {

#ifdef LLMHUB_SNAPDRAGON
constexpr const char * TAG = "LlamaCppStreamNpu";
#else
constexpr const char * TAG = "LlamaCppStream";
#endif
constexpr int kDecodeThreads = 2;

std::recursive_mutex g_mutex;                // load, unload and start/stop
std::unique_ptr<bmoe::Session> g_session;    // the streaming engine
llama_model * g_template_model = nullptr;     // vocab-only load of the same file, for its chat template
std::atomic<int> g_epoch{0};

std::thread g_worker;
std::mutex g_queue_mutex;
std::condition_variable g_queue_cv;
std::deque<std::string> g_pieces;
bool g_done = true;
std::atomic<bool> g_stop{false};
std::string g_utf8_buffer;
double g_decode_tps = 0.0;

std::string from_jstring(JNIEnv * env, jstring value) {
    if (value == nullptr) return {};
    const char * chars = env->GetStringUTFChars(value, nullptr);
    std::string result = chars != nullptr ? chars : "";
    if (chars != nullptr) env->ReleaseStringUTFChars(value, chars);
    return result;
}

bool valid_utf8(const std::string & value) {
    const auto * p = reinterpret_cast<const unsigned char *>(value.data());
    const auto * end = p + value.size();
    while (p < end) {
        int count = 0;
        if ((*p & 0x80) == 0) count = 1;
        else if ((*p & 0xE0) == 0xC0) count = 2;
        else if ((*p & 0xF0) == 0xE0) count = 3;
        else if ((*p & 0xF8) == 0xF0) count = 4;
        else return false;
        if (p + count > end) return false;
        for (int i = 1; i < count; ++i) {
            if ((p[i] & 0xC0) != 0x80) return false;
        }
        p += count;
    }
    return true;
}

void bmoe_log_sink(bmoe::LogLevel level, const char * message, void *) {
    const int priority = level == bmoe::LogLevel::Error ? ANDROID_LOG_ERROR
                        : level == bmoe::LogLevel::Warn ? ANDROID_LOG_WARN : ANDROID_LOG_INFO;
    __android_log_write(priority, TAG, message);
}

// Ends the running generation, if any, and waits for its thread.
void stop_worker() {
    g_stop.store(true);
    if (g_session) g_session->cancel();
    if (g_worker.joinable()) g_worker.join();
    std::lock_guard<std::mutex> lock(g_queue_mutex);
    g_pieces.clear();
    g_done = true;
    g_utf8_buffer.clear();
}

void unload_all() {
    stop_worker();
    g_session.reset();
    if (g_template_model != nullptr) {
        llama_model_free(g_template_model);
        g_template_model = nullptr;
    }
    g_epoch.fetch_add(1);
}

std::string mime_for(const std::string & path) {
    std::string lower = path;
    std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) { return std::tolower(c); });
    if (lower.size() >= 4 && lower.compare(lower.size() - 4, 4, ".png") == 0) return "image/png";
    if (lower.size() >= 5 && lower.compare(lower.size() - 5, 5, ".webp") == 0) return "image/webp";
    return "image/jpeg";
}

} // namespace

#ifdef LLMHUB_SNAPDRAGON
void llama_log_sink(ggml_log_level level, const char * text, void *) {
    if (level == GGML_LOG_LEVEL_DEBUG) return;
    const int priority = level == GGML_LOG_LEVEL_ERROR ? ANDROID_LOG_ERROR
                        : level == GGML_LOG_LEVEL_WARN ? ANDROID_LOG_WARN : ANDROID_LOG_INFO;
    __android_log_write(priority, TAG, text);
}
bool g_backend_ready = false;
#endif

// CPU build: LlamaCppStreamNative.nativeInit calls LlamaCppNative.nativeInit first, which registers the CPU backend
// variant in the shared libllmhub_ggml; this only routes the engine's own logs. Snapdragon build: loads the release's
// backends (CPU, OpenCL, Hexagon) from the native library folder, with the Hexagon skels in [htp_dir].
extern "C" JNIEXPORT jint JNICALL
STREAM_JNI(nativeInitStream)(JNIEnv * env, jobject, jstring library_dir, jstring htp_dir) {
    bmoe::set_log_sink(bmoe_log_sink);
#ifdef LLMHUB_SNAPDRAGON
    std::lock_guard<std::recursive_mutex> lock(g_mutex);
    if (!g_backend_ready) {
        llama_log_set(llama_log_sink, nullptr);
        const std::string htp_path = from_jstring(env, htp_dir);
        if (!htp_path.empty()) {
            const char * old_path = std::getenv("ADSP_LIBRARY_PATH");
            const std::string path = htp_path + (old_path != nullptr ? ";" + std::string(old_path) : "");
            setenv("ADSP_LIBRARY_PATH", path.c_str(), 1);
        }
        ggml_backend_load_all_from_path(from_jstring(env, library_dir).c_str());
        llama_backend_init();
        g_backend_ready = true;
    }
#else
    (void) env;
    (void) library_dir;
    (void) htp_dir;
#endif
    return 0;
}

extern "C" JNIEXPORT jint JNICALL
STREAM_JNI(nativeLoadModel)(
    JNIEnv * env, jobject, jstring model_path, jstring mmproj_path,
    jint context_size, jint thread_count, jstring device_name, jint, jint) {
    std::lock_guard<std::recursive_mutex> lock(g_mutex);
    unload_all();
    const std::string device = from_jstring(env, device_name);

    bmoe::SessionConfig cfg;
    cfg.model_path = from_jstring(env, model_path);
    cfg.mmproj = from_jstring(env, mmproj_path);
    cfg.n_ctx = std::max(512, static_cast<int>(context_size));
    // A streamed decode mostly waits on storage and spreads little work per token, so a few threads decode faster
    // than many (measured on the Galaxy Tab S9+: 2 threads 244 ms of compute per token, 6 threads 669 ms). Prompt
    // reading is compute-bound and keeps the app's usual count.
    cfg.n_threads = std::min(kDecodeThreads, std::max(1, static_cast<int>(thread_count)));
    cfg.n_threads_batch = std::max(1, static_cast<int>(thread_count));
    cfg.n_batch = 512;
    cfg.n_ubatch = 512;
    cfg.chatml = false;                 // the app formats the chat itself and passes the finished prompt
    cfg.n_gpu_layers = 0;
    cfg.device_use = bmoe::DeviceUse::CpuOnly;
    cfg.moe.enabled = true;
    cfg.moe.cache_auto = true;          // size the expert cache from the free RAM, leaving cache_floor_mb free
    // ...but at most 30% of the device's RAM. Android vendors kill an app whose own memory passes a share of the
    // RAM however much is free (Samsung's limit on the 11 GB Galaxy Tab S9+ is 6 GiB, which a 5.8 GB cache plus
    // the resident weights crossed), so the free RAM alone is not a safe budget.
    const long long total_mb = static_cast<long long>(sysconf(_SC_PHYS_PAGES)) * sysconf(_SC_PAGESIZE) / (1024 * 1024);
    cfg.moe.cache_ceil_mb = static_cast<int>(std::max<long long>(1536, total_mb * 3 / 10));
    cfg.moe.dense_weights = bmoe::DenseWeightsMode::Pinned;
    cfg.moe.io_threads = 4;
    cfg.moe.o_direct = true;
    // An accelerator (Hexagon NPU "HTP0" or Vulkan "Vulkan0") reads the prompt; decode stays on the CPU, where a
    // one-token step is faster (dense layers on the device read the prompt at a tenth of the speed, the experts then
    // running on the CPU). The slots it reads experts through are held only while a prompt is read: the NPU can map
    // only about 3 GB, and holding them for the session ran out of that space. A device that cannot open, or cannot
    // multiply by the model's expert type (the NPU and Gemma 4's IQ2_XS experts), fails the load, and the app then
    // streams on the CPU.
    if (!device.empty()) {
        cfg.moe.dense_weights = bmoe::DenseWeightsMode::Anonymous;
        cfg.n_batch = 2048;
        cfg.n_ubatch = 2048;
        cfg.prefill.device = device;
        cfg.prefill.slots_on_demand = true;
        cfg.prefill.verify = false;
        cfg.prefill.best_effort = false;
    }

    std::string error;
    // The engine reads the experts itself and hands the page cache back after load: llama.cpp's read-ahead of the whole
    // file (14 GB for Gemma 4 26B) would fill a 12 GB phone first and get the app killed mid-load (llama-model.cpp).
    setenv("LLMHUB_NO_MMAP_PREFETCH", "1", 1);
    g_session = bmoe::Session::open(cfg, error);
    unsetenv("LLMHUB_NO_MMAP_PREFETCH");
    if (!g_session) {
        __android_log_print(ANDROID_LOG_ERROR, TAG, "Flash streaming load failed: %s", error.c_str());
        return 1;
    }

    llama_model_params params = llama_model_default_params();
    params.vocab_only = true;
    ggml_backend_dev_t no_devices[] = {nullptr};
    params.devices = no_devices;
    g_template_model = llama_model_load_from_file(cfg.model_path.c_str(), params);
    if (g_template_model == nullptr) {
        __android_log_write(ANDROID_LOG_WARN, TAG, "Vocab-only load for the chat template failed");
    }
    __android_log_print(ANDROID_LOG_INFO, TAG, "Flash streaming loaded %s (%s) in %.1f s, vision=%d",
                        cfg.model_path.c_str(), g_session->arch().c_str(), g_session->load_seconds(),
                        g_session->vision() ? 1 : 0);
    return 0;
}

extern "C" JNIEXPORT jboolean JNICALL
STREAM_JNI(nativeSupportsVision)(JNIEnv *, jobject) {
    std::lock_guard<std::recursive_mutex> lock(g_mutex);
    return g_session && g_session->vision() ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jstring JNICALL
STREAM_JNI(nativeMediaMarker)(JNIEnv * env, jobject) {
    return env->NewStringUTF(bmoe::image_marker());
}

extern "C" JNIEXPORT jstring JNICALL
STREAM_JNI(nativeFormatChat)(JNIEnv * env, jobject, jobjectArray role_array, jobjectArray content_array) {
    std::lock_guard<std::recursive_mutex> lock(g_mutex);
    if (g_template_model == nullptr || role_array == nullptr || content_array == nullptr) return nullptr;
    const jsize count = env->GetArrayLength(role_array);
    if (count == 0 || count != env->GetArrayLength(content_array)) return nullptr;
    std::vector<std::string> roles, contents;
    for (jsize i = 0; i < count; ++i) {
        auto role = static_cast<jstring>(env->GetObjectArrayElement(role_array, i));
        auto content = static_cast<jstring>(env->GetObjectArrayElement(content_array, i));
        roles.push_back(from_jstring(env, role));
        contents.push_back(from_jstring(env, content));
        env->DeleteLocalRef(role);
        env->DeleteLocalRef(content);
    }
    std::vector<llama_chat_message> messages;
    for (jsize i = 0; i < count; ++i) messages.push_back({roles[i].c_str(), contents[i].c_str()});
    const char * tmpl = llama_model_chat_template(g_template_model, nullptr);
    const int32_t required = llama_chat_apply_template(tmpl, messages.data(), messages.size(), true, nullptr, 0);
    if (required < 0) return nullptr;
    std::vector<char> buffer(static_cast<size_t>(required) + 1);
    const int32_t written = llama_chat_apply_template(
        tmpl, messages.data(), messages.size(), true, buffer.data(), static_cast<int32_t>(buffer.size()));
    if (written < 0) return nullptr;
    return env->NewStringUTF(std::string(buffer.data(), static_cast<size_t>(written)).c_str());
}

extern "C" JNIEXPORT jint JNICALL
STREAM_JNI(nativeStartCompletion)(
    JNIEnv * env, jobject, jstring formatted_prompt, jobjectArray image_path_array,
    jint max_tokens, jfloat temperature, jint top_k, jfloat top_p, jstring grammar) {
    std::lock_guard<std::recursive_mutex> lock(g_mutex);
    if (!g_session) return 1;
    stop_worker();

    auto req = std::make_shared<bmoe::GenerateRequest>();
    req->prompt = from_jstring(env, formatted_prompt);
    if (req->prompt.empty()) return 2;
    req->n_predict = std::max(1, static_cast<int>(max_tokens));
    req->clear_kv = true;               // like the other runtimes: each completion carries the whole conversation
    req->render_text = false;           // only the per-token pieces are read
    req->stop = &g_stop;
    bmoe::RequestSampling sampling;
    sampling.temp = std::max(0.0f, static_cast<float>(temperature));
    sampling.top_k = std::max(1, static_cast<int>(top_k));
    sampling.top_p = std::clamp(static_cast<float>(top_p), 0.0f, 1.0f);
    req->sampling = sampling;
    if (!from_jstring(env, grammar).empty()) {
        __android_log_write(ANDROID_LOG_WARN, TAG, "Output grammar is not supported with flash streaming; generating without it");
    }

    const jsize image_count = image_path_array == nullptr ? 0 : env->GetArrayLength(image_path_array);
    if (image_count > 0 && !g_session->vision()) return 6;
    for (jsize i = 0; i < image_count; ++i) {
        auto path_value = static_cast<jstring>(env->GetObjectArrayElement(image_path_array, i));
        const std::string path = from_jstring(env, path_value);
        env->DeleteLocalRef(path_value);
        std::ifstream file(path, std::ios::binary);
        if (!file) return 7;
        bmoe::ChatImage image;
        image.bytes.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
        image.mime = mime_for(path);
        req->images.push_back(std::move(image));
    }

    {
        std::lock_guard<std::mutex> lock_queue(g_queue_mutex);
        g_pieces.clear();
        g_done = false;
        g_utf8_buffer.clear();
    }
    g_stop.store(false);
    g_decode_tps = 0.0;
    bmoe::Session * session = g_session.get();
    g_worker = std::thread([session, req]() {
        bmoe::RunResult result;
        // A backend can throw from inside a graph (ggml-hexagon on a failed NPU buffer mapping); uncaught, that ends
        // the whole app. It ends this reply instead.
        try {
            result = session->generate(*req, [](const bmoe::TokenMetrics & metrics) {
                if (metrics.piece.empty()) return;
                std::lock_guard<std::mutex> lock(g_queue_mutex);
                g_pieces.push_back(metrics.piece);
                g_queue_cv.notify_one();
            });
        } catch (const std::exception & e) {
            result.ok = false;
            result.error = e.what();
        } catch (...) {
            result.ok = false;
            result.error = "unknown exception";
        }
        if (!result.ok) {
            __android_log_print(ANDROID_LOG_ERROR, TAG, "Flash streaming generation failed: %s", result.error.c_str());
        }
        if (result.summary.gen_seconds > 0.0 && result.summary.n_generated > 0) {
            g_decode_tps = result.summary.n_generated / result.summary.gen_seconds;
        }
        const auto & sum = result.summary;
        __android_log_print(ANDROID_LOG_INFO, TAG,
                            "Turn: prompt %d tokens in %.2f s (%.1f tok/s), %d tokens generated at %.2f tok/s",
                            sum.n_prompt, sum.prefill_seconds,
                            sum.prefill_seconds > 0.0 ? sum.n_prompt / sum.prefill_seconds : 0.0,
                            sum.n_generated, g_decode_tps);
        std::lock_guard<std::mutex> lock(g_queue_mutex);
        g_done = true;
        g_queue_cv.notify_all();
    });
    return 0;
}

extern "C" JNIEXPORT jstring JNICALL
STREAM_JNI(nativeNextToken)(JNIEnv * env, jobject) {
    std::unique_lock<std::mutex> lock(g_queue_mutex);
    g_queue_cv.wait(lock, [] { return !g_pieces.empty() || g_done; });
    if (g_pieces.empty()) {
        if (g_utf8_buffer.empty()) return nullptr;
        std::string rest;
        rest.swap(g_utf8_buffer);
        return env->NewStringUTF(rest.c_str());
    }
    g_utf8_buffer += g_pieces.front();
    g_pieces.pop_front();
    if (!valid_utf8(g_utf8_buffer)) return env->NewStringUTF("");
    std::string out;
    out.swap(g_utf8_buffer);
    return env->NewStringUTF(out.c_str());
}

extern "C" JNIEXPORT void JNICALL
STREAM_JNI(nativeStop)(JNIEnv *, jobject) {
    std::lock_guard<std::recursive_mutex> lock(g_mutex);
    stop_worker();
}

extern "C" JNIEXPORT void JNICALL
STREAM_JNI(nativeReset)(JNIEnv *, jobject) {
    std::lock_guard<std::recursive_mutex> lock(g_mutex);
    stop_worker();
}

extern "C" JNIEXPORT void JNICALL
STREAM_JNI(nativeUnload)(JNIEnv *, jobject) {
    std::lock_guard<std::recursive_mutex> lock(g_mutex);
    unload_all();
}

extern "C" JNIEXPORT jdouble JNICALL
STREAM_JNI(nativeDecodeSpeed)(JNIEnv *, jobject) {
    return g_decode_tps;
}

extern "C" JNIEXPORT jint JNICALL
STREAM_JNI(nativeModelEpoch)(JNIEnv *, jobject) {
    return g_epoch.load();
}
