#include <jni.h>
#include <android/log.h>

#include <string>
#include <vector>
#include <thread>
#include <chrono>
#include <cstring>

#include "llama.h"

#define TAG "LlmHubJni"

// Persistent native objects --------------------------------------------------
static llama_model  * g_model  = nullptr;
static const llama_vocab * g_vocab = nullptr;
static llama_context * g_ctx   = nullptr;

// ---------------------------------------------------------------------------
extern "C" JNIEXPORT jint JNICALL
Java_com_example_llmhub_inference_LlamaInferenceService_initLlama(
        JNIEnv * env,
        jobject /* this */,
        jstring modelPath) {
    const char * path = env->GetStringUTFChars(modelPath, nullptr);
    __android_log_print(ANDROID_LOG_INFO, TAG, "Loading model from: %s", path);

    // Initialise backend (loads OpenCL etc.)
    llama_backend_init();

    // Forward llama.cpp internal logs to logcat
    llama_log_set([](ggml_log_level /*level*/, const char * msg, void *) {
        __android_log_print(ANDROID_LOG_INFO, TAG, "%s", msg);
    }, nullptr);

    // ---------------- Model ----------------
    llama_model_params mparams = llama_model_default_params();
    // Offload as many layers as possible to the first GPU (OpenCL)
    mparams.n_gpu_layers = 99;  // 99 = "all that fit"
    g_model = llama_model_load_from_file(path, mparams);
    env->ReleaseStringUTFChars(modelPath, path);

    if (!g_model) {
        __android_log_print(ANDROID_LOG_ERROR, TAG, "Failed to load model");
        return -1;
    }

    g_vocab = llama_model_get_vocab(g_model);

    // ---------------- Context -------------
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx   = 2048;   // context length
    cparams.n_batch = 512;    // max tokens per decode call
    cparams.no_perf = false;  // keep perf counters enabled

    g_ctx = llama_init_from_model(g_model, cparams);
    if (!g_ctx) {
        __android_log_print(ANDROID_LOG_ERROR, TAG, "Failed to create context");
        llama_model_free(g_model);
        g_model = nullptr;
        g_vocab = nullptr;
        return -2;
    }

    // Use all available cores
    llama_set_n_threads(g_ctx, std::thread::hardware_concurrency(), std::thread::hardware_concurrency());

    return 0;
}

// ---------------------------------------------------------------------------
extern "C" JNIEXPORT jstring JNICALL
Java_com_example_llmhub_inference_LlamaInferenceService_generateResponse(
        JNIEnv * env,
        jobject /* this */,
        jstring prompt) {
    if (!g_ctx || !g_vocab) {
        return env->NewStringUTF("Model not initialized");
    }

    const char * prompt_chars = env->GetStringUTFChars(prompt, nullptr);

    // Build Gemma-style dialogue template
    std::string full_prompt = std::string("<start_of_turn>user\n") +
                               prompt_chars +
                               "\n<end_of_turn>\n<start_of_turn>model\n";
    env->ReleaseStringUTFChars(prompt, prompt_chars);

    // ---------------- Tokenize ------------
    const int n_prompt = -llama_tokenize(g_vocab, full_prompt.c_str(), (int)full_prompt.size(),
                                         /*tokens=*/nullptr, /*n_tokens_max=*/0,
                                         /*add_special=*/true, /*parse_special=*/true);
    if (n_prompt <= 0) {
        return env->NewStringUTF("Tokenization failed");
    }

    std::vector<llama_token> prompt_tokens(n_prompt);
    if (llama_tokenize(g_vocab, full_prompt.c_str(), (int)full_prompt.size(),
                       prompt_tokens.data(), prompt_tokens.size(),
                       /*add_special=*/true, /*parse_special=*/true) < 0) {
        return env->NewStringUTF("Tokenization failed");
    }

    // ---------------- Sampler -------------
    llama_sampler_chain_params schain_params = llama_sampler_chain_default_params();
    schain_params.no_perf = false;
    llama_sampler * sampler_chain = llama_sampler_chain_init(schain_params);
    llama_sampler_chain_add(sampler_chain, llama_sampler_init_greedy());

    // ---------------- Generation ----------
    std::string result;

    auto t_start = std::chrono::steady_clock::now();
    __android_log_print(ANDROID_LOG_INFO, TAG, "Starting generation: prompt tokens=%d", n_prompt);

    // First batch contains the full prompt
    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), (int32_t)prompt_tokens.size());

    const int n_predict = 64; // max tokens to generate; reduce for faster response on mobile
    int n_pos = 0;

    while (n_pos + batch.n_tokens < n_prompt + n_predict) {
        auto t0 = std::chrono::steady_clock::now();
        if (llama_decode(g_ctx, batch) != 0) {
            __android_log_print(ANDROID_LOG_ERROR, TAG, "llama_decode failed");
            break;
        }
        auto t1 = std::chrono::steady_clock::now();
        double ms_decode = std::chrono::duration<double, std::milli>(t1 - t0).count();
        __android_log_print(ANDROID_LOG_VERBOSE, TAG, "decode step took %.2f ms (pos=%d)", ms_decode, n_pos + batch.n_tokens);

        n_pos += batch.n_tokens;

        // Sample next token (greedy)
        llama_token token_id = llama_sampler_sample(sampler_chain, g_ctx, /*idx=*/-1);

        if (llama_vocab_is_eog(g_vocab, token_id)) {
            break; // end-of-generation token reached
        }

        char buf[128];
        int n = llama_token_to_piece(g_vocab, token_id, buf, sizeof(buf), /*lstrip=*/0, /*special=*/true);
        if (n > 0) {
            buf[n] = '\0'; // ensure null-termination for logcat
            // Stop early if end-of-turn marker produced
            if (strncmp(buf, "<end_of_turn>", 13) == 0) {
                __android_log_print(ANDROID_LOG_VERBOSE, TAG, "End-of-turn token reached, stopping early");
                break;
            }

            result.append(buf, n);
            __android_log_print(ANDROID_LOG_VERBOSE, TAG, "token[%d]='%s'", (int)result.size(), buf);

            // Forward token to Kotlin layer for streaming UI
            jclass cls = env->FindClass("com/example/llmhub/inference/LlamaInferenceService");
            if (cls != nullptr) {
                jmethodID mid = env->GetStaticMethodID(cls, "onNativeToken", "(Ljava/lang/String;)V");
                if (mid != nullptr) {
                    jstring jtok = env->NewStringUTF(buf);
                    env->CallStaticVoidMethod(cls, mid, jtok);
                    env->DeleteLocalRef(jtok);
                }
                env->DeleteLocalRef(cls);
            }
        }

        // Prepare batch for the newly generated token
        batch = llama_batch_get_one(&token_id, 1);
    }

    llama_sampler_free(sampler_chain);

    auto t_end = std::chrono::steady_clock::now();
    double ms_total = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    __android_log_print(ANDROID_LOG_INFO, TAG, "Generation finished: %zu bytes, %.2f s", result.size(), ms_total / 1000.0);

    // Print internal timing statistics from llama.cpp
    llama_perf_context_print(g_ctx);

    return env->NewStringUTF(result.c_str());
}

// ---------------------------------------------------------------------------
extern "C" JNIEXPORT void JNICALL
Java_com_example_llmhub_inference_LlamaInferenceService_releaseLlama(
        JNIEnv * /*env*/,
        jobject /* this */) {
    if (g_ctx) {
        llama_free(g_ctx);
        g_ctx = nullptr;
    }
    if (g_model) {
        llama_model_free(g_model);
        g_model = nullptr;
        g_vocab = nullptr;
    }
    llama_backend_free();
}