#include <jni.h>
#include <android/log.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <string>
#include <thread>
#include <vector>

#include "llama.h"

namespace {

constexpr const char * TAG = "LlamaCppFallback";

llama_model * g_model = nullptr;
llama_context * g_context = nullptr;
llama_sampler * g_sampler = nullptr;
std::atomic_bool g_stop{false};
int32_t g_generated = 0;
int32_t g_max_tokens = 0;
int64_t g_decode_start_us = 0;
std::string g_utf8_buffer;
bool g_backend_initialized = false;

void log_callback(ggml_log_level level, const char * text, void *) {
    int priority = ANDROID_LOG_DEBUG;
    if (level == GGML_LOG_LEVEL_ERROR) priority = ANDROID_LOG_ERROR;
    else if (level == GGML_LOG_LEVEL_WARN) priority = ANDROID_LOG_WARN;
    else if (level == GGML_LOG_LEVEL_INFO) priority = ANDROID_LOG_INFO;
    __android_log_write(priority, TAG, text);
}

int64_t now_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
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

std::string from_jstring(JNIEnv * env, jstring value) {
    if (value == nullptr) return {};
    const char * chars = env->GetStringUTFChars(value, nullptr);
    std::string result(chars != nullptr ? chars : "");
    if (chars != nullptr) env->ReleaseStringUTFChars(value, chars);
    return result;
}

void free_sampler() {
    if (g_sampler != nullptr) {
        llama_sampler_free(g_sampler);
        g_sampler = nullptr;
    }
}

void unload_model() {
    free_sampler();
    if (g_context != nullptr) {
        llama_free(g_context);
        g_context = nullptr;
    }
    if (g_model != nullptr) {
        llama_model_free(g_model);
        g_model = nullptr;
    }
    g_utf8_buffer.clear();
    g_generated = 0;
    g_max_tokens = 0;
}

std::string format_chat(
    const std::vector<std::string> & roles,
    const std::vector<std::string> & contents
) {
    std::vector<llama_chat_message> messages;
    messages.reserve(roles.size());
    for (size_t i = 0; i < roles.size(); ++i) {
        messages.push_back({roles[i].c_str(), contents[i].c_str()});
    }

    const char * chat_template = llama_model_chat_template(g_model, nullptr);
    int32_t required = llama_chat_apply_template(
        chat_template, messages.data(), messages.size(), true, nullptr, 0);
    if (required >= 0) {
        std::vector<char> buffer(static_cast<size_t>(required) + 1);
        int32_t written = llama_chat_apply_template(
            chat_template, messages.data(), messages.size(), true,
            buffer.data(), static_cast<int32_t>(buffer.size()));
        if (written >= 0) return std::string(buffer.data(), static_cast<size_t>(written));
    }

    __android_log_write(ANDROID_LOG_WARN, TAG, "Model chat template could not be applied");
    return {};
}

bool read_messages(
    JNIEnv * env,
    jobjectArray role_array,
    jobjectArray content_array,
    std::vector<std::string> & roles,
    std::vector<std::string> & contents
) {
    const jsize role_count = env->GetArrayLength(role_array);
    const jsize content_count = env->GetArrayLength(content_array);
    if (role_count == 0 || role_count != content_count) return false;
    roles.reserve(role_count);
    contents.reserve(content_count);
    for (jsize i = 0; i < role_count; ++i) {
        auto role = static_cast<jstring>(env->GetObjectArrayElement(role_array, i));
        auto content = static_cast<jstring>(env->GetObjectArrayElement(content_array, i));
        roles.push_back(from_jstring(env, role));
        contents.push_back(from_jstring(env, content));
        env->DeleteLocalRef(role);
        env->DeleteLocalRef(content);
    }
    return true;
}

int decode_tokens(const std::vector<llama_token> & tokens) {
    const int32_t batch_size = static_cast<int32_t>(llama_n_batch(g_context));
    for (size_t offset = 0; offset < tokens.size(); offset += batch_size) {
        if (g_stop.load()) return 2;
        const int32_t count = std::min<int32_t>(
            batch_size, static_cast<int32_t>(tokens.size() - offset));
        llama_batch batch = llama_batch_get_one(
            const_cast<llama_token *>(tokens.data() + offset), count);
        const int result = llama_decode(g_context, batch);
        if (result != 0) return result;
    }
    return 0;
}

} // namespace

extern "C" JNIEXPORT jint JNICALL
Java_com_llmhub_llmhub_inference_LlamaCppNative_nativeInit(JNIEnv *, jobject) {
    if (!g_backend_initialized) {
        llama_log_set(log_callback, nullptr);
        llama_backend_init();
        g_backend_initialized = true;
    }
    return 0;
}

extern "C" JNIEXPORT jint JNICALL
Java_com_llmhub_llmhub_inference_LlamaCppNative_nativeLoadModel(
    JNIEnv * env, jobject, jstring model_path, jint context_size, jint thread_count) {
    unload_model();
    const std::string path = from_jstring(env, model_path);

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    g_model = llama_model_load_from_file(path.c_str(), model_params);
    if (g_model == nullptr) return 1;

    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = static_cast<uint32_t>(context_size);
    context_params.n_batch = static_cast<uint32_t>(std::min(context_size, 512));
    context_params.n_ubatch = context_params.n_batch;
    context_params.n_threads = thread_count;
    context_params.n_threads_batch = thread_count;
    g_context = llama_init_from_model(g_model, context_params);
    if (g_context == nullptr) {
        unload_model();
        return 2;
    }
    return 0;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_llmhub_llmhub_inference_LlamaCppNative_nativeFormatChat(
    JNIEnv * env, jobject, jobjectArray role_array, jobjectArray content_array) {
    if (g_model == nullptr) return nullptr;
    std::vector<std::string> roles;
    std::vector<std::string> contents;
    if (!read_messages(env, role_array, content_array, roles, contents)) return nullptr;
    const std::string formatted = format_chat(roles, contents);
    if (formatted.empty()) return nullptr;
    return env->NewStringUTF(formatted.c_str());
}

extern "C" JNIEXPORT jint JNICALL
Java_com_llmhub_llmhub_inference_LlamaCppNative_nativeStartCompletion(
    JNIEnv * env, jobject, jstring formatted_prompt,
    jint max_tokens, jfloat temperature, jint top_k, jfloat top_p) {
    if (g_model == nullptr || g_context == nullptr) return 1;

    g_stop.store(false);
    g_generated = 0;
    const int32_t requested_max_tokens = std::max(1, static_cast<int32_t>(max_tokens));
    g_utf8_buffer.clear();
    llama_memory_clear(llama_get_memory(g_context), false);

    free_sampler();
    g_sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(g_sampler, llama_sampler_init_top_k(std::max(1, static_cast<int32_t>(top_k))));
    llama_sampler_chain_add(g_sampler, llama_sampler_init_top_p(
        std::clamp(static_cast<float>(top_p), 0.0f, 1.0f), 1));
    llama_sampler_chain_add(g_sampler, llama_sampler_init_temp(
        std::max(0.0f, static_cast<float>(temperature))));
    llama_sampler_chain_add(g_sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    const std::string prompt = from_jstring(env, formatted_prompt);
    if (prompt.empty()) return 2;
    const llama_vocab * vocab = llama_model_get_vocab(g_model);
    int32_t token_count = llama_tokenize(
        vocab, prompt.data(), static_cast<int32_t>(prompt.size()), nullptr, 0, true, true);
    if (token_count >= 0) return 3;
    token_count = -token_count;
    std::vector<llama_token> tokens(static_cast<size_t>(token_count));
    token_count = llama_tokenize(
        vocab, prompt.data(), static_cast<int32_t>(prompt.size()),
        tokens.data(), static_cast<int32_t>(tokens.size()), true, true);
    if (token_count < 0) return 4;
    tokens.resize(static_cast<size_t>(token_count));

    const int32_t context_size = static_cast<int32_t>(llama_n_ctx(g_context));
    // Preserve the conversation first. The requested response length is only a ceiling; it must
    // not evict chat history before generation has even started. Keep one context slot for the
    // first generated token and cap the response to whatever remains after prompt evaluation.
    const int32_t prompt_limit = std::max(1, context_size - 1);
    if (static_cast<int32_t>(tokens.size()) > prompt_limit) {
        const size_t remove_count = tokens.size() - static_cast<size_t>(prompt_limit);
        const llama_token first = tokens.front();
        tokens.erase(tokens.begin(), tokens.begin() + static_cast<std::ptrdiff_t>(remove_count));
        // Preserve the tokenizer's leading special token while retaining the newest context.
        if (!tokens.empty()) tokens.front() = first;
        __android_log_write(ANDROID_LOG_WARN, TAG, "Prompt truncated to fit CPU fallback context");
    }

    g_max_tokens = std::max(
        1,
        std::min(requested_max_tokens, context_size - static_cast<int32_t>(tokens.size())));
    __android_log_print(
        ANDROID_LOG_DEBUG,
        TAG,
        "Starting completion with %d prompt tokens and room for %d/%d requested output tokens",
        static_cast<int32_t>(tokens.size()),
        g_max_tokens,
        requested_max_tokens);

    const int decode_result = decode_tokens(tokens);
    if (decode_result != 0) return 5;
    g_decode_start_us = now_us();
    return 0;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_llmhub_llmhub_inference_LlamaCppNative_nativeNextToken(JNIEnv * env, jobject) {
    if (g_context == nullptr || g_sampler == nullptr || g_stop.load() || g_generated >= g_max_tokens) {
        return nullptr;
    }

    llama_token token = llama_sampler_sample(g_sampler, g_context, -1);
    const llama_vocab * vocab = llama_model_get_vocab(g_model);
    if (llama_vocab_is_eog(vocab, token)) return nullptr;

    int32_t piece_size = llama_token_to_piece(vocab, token, nullptr, 0, 0, true);
    if (piece_size < 0) piece_size = -piece_size;
    std::vector<char> piece(static_cast<size_t>(piece_size) + 1);
    const int32_t written = llama_token_to_piece(
        vocab, token, piece.data(), static_cast<int32_t>(piece.size()), 0, true);
    if (written > 0) g_utf8_buffer.append(piece.data(), static_cast<size_t>(written));

    llama_batch batch = llama_batch_get_one(&token, 1);
    if (llama_decode(g_context, batch) != 0) return nullptr;
    ++g_generated;

    if (!valid_utf8(g_utf8_buffer)) return env->NewStringUTF("");
    jstring result = env->NewStringUTF(g_utf8_buffer.c_str());
    g_utf8_buffer.clear();
    return result;
}

extern "C" JNIEXPORT void JNICALL
Java_com_llmhub_llmhub_inference_LlamaCppNative_nativeStop(JNIEnv *, jobject) {
    g_stop.store(true);
}

extern "C" JNIEXPORT void JNICALL
Java_com_llmhub_llmhub_inference_LlamaCppNative_nativeReset(JNIEnv *, jobject) {
    g_stop.store(true);
    if (g_context != nullptr) llama_memory_clear(llama_get_memory(g_context), false);
    if (g_sampler != nullptr) llama_sampler_reset(g_sampler);
}

extern "C" JNIEXPORT void JNICALL
Java_com_llmhub_llmhub_inference_LlamaCppNative_nativeUnload(JNIEnv *, jobject) {
    g_stop.store(true);
    unload_model();
}

extern "C" JNIEXPORT jdouble JNICALL
Java_com_llmhub_llmhub_inference_LlamaCppNative_nativeDecodeSpeed(JNIEnv *, jobject) {
    const int64_t elapsed = now_us() - g_decode_start_us;
    return elapsed > 0 ? static_cast<double>(g_generated) * 1000000.0 / static_cast<double>(elapsed) : 0.0;
}
