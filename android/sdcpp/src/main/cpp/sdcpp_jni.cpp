// JNI bridge for LTX-Video generation through stable-diffusion.cpp (Vulkan / CPU ggml backends).
#include <android/log.h>
#include <android/set_abort_message.h>
#include <jni.h>
#include <sys/system_properties.h>

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#include "ggml.h"
#include "stable-diffusion.h"

#define LOG_TAG "SdCppJni"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace {

JavaVM* g_vm = nullptr;

struct CallbackTarget {
    jobject callback       = nullptr;  // global ref, valid only while a generation runs
    jmethodID on_progress  = nullptr;
    jmethodID on_stage     = nullptr;
};

std::mutex g_callback_mutex;
CallbackTarget g_target;
std::string g_last_error;
// stable-diffusion.cpp reports tensor-loading progress through the same callback; only steps
// that match the requested sampling step count are denoising progress.
std::atomic<int> g_expected_steps{0};

JNIEnv* attached_env(bool* attached) {
    *attached = false;
    if (g_vm == nullptr) {
        return nullptr;
    }
    JNIEnv* env = nullptr;
    if (g_vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6) == JNI_OK) {
        return env;
    }
    if (g_vm->AttachCurrentThread(&env, nullptr) == JNI_OK) {
        *attached = true;
        return env;
    }
    return nullptr;
}

void emit_stage(int stage) {
    std::lock_guard<std::mutex> lock(g_callback_mutex);
    if (g_target.callback == nullptr) {
        return;
    }
    bool attached = false;
    JNIEnv* env   = attached_env(&attached);
    if (env != nullptr) {
        env->CallVoidMethod(g_target.callback, g_target.on_stage, stage);
        if (env->ExceptionCheck()) {
            env->ExceptionClear();
        }
    }
    if (attached) {
        g_vm->DetachCurrentThread();
    }
}

// Stage ids shared with SdCppNative.kt.
constexpr int STAGE_ENCODING_PROMPT = 1;
constexpr int STAGE_DENOISING       = 2;
constexpr int STAGE_DECODING        = 3;

void on_sd_log(enum sd_log_level_t level, const char* text, void*) {
    if (text == nullptr) {
        return;
    }
    switch (level) {
        case SD_LOG_ERROR:
            LOGE("%s", text);
            g_last_error = text;
            break;
        case SD_LOG_WARN:
            LOGW("%s", text);
            break;
        case SD_LOG_INFO:
            LOGI("%s", text);
            break;
        default:
            break;
    }
    if (strstr(text, "sampling completed") != nullptr || strstr(text, "decoding") != nullptr) {
        emit_stage(STAGE_DECODING);
    }
}

void on_sd_progress(int step, int steps, float, void*) {
    if (steps != g_expected_steps.load()) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_callback_mutex);
    if (g_target.callback == nullptr) {
        return;
    }
    bool attached = false;
    JNIEnv* env   = attached_env(&attached);
    if (env != nullptr) {
        env->CallVoidMethod(g_target.callback, g_target.on_progress, step, steps);
        if (env->ExceptionCheck()) {
            env->ExceptionClear();
        }
    }
    if (attached) {
        g_vm->DetachCurrentThread();
    }
}

std::string to_string(JNIEnv* env, jstring value) {
    if (value == nullptr) {
        return {};
    }
    const char* chars = env->GetStringUTFChars(value, nullptr);
    std::string out   = chars != nullptr ? chars : "";
    if (chars != nullptr) {
        env->ReleaseStringUTFChars(value, chars);
    }
    return out;
}

struct Session {
    sd_ctx_t* ctx = nullptr;
    std::atomic<bool> cancelled{false};
};

}  // namespace

// Diagnostics: `adb shell setprop debug.llmhub.ggml_vk_env "GGML_VK_DISABLE_COOPMAT=1;GGML_VK_DISABLE_F16=1"`
// sets ggml-vulkan switches before the backend initializes, to find a broken driver feature.
static void apply_debug_ggml_env() {
    char spec[PROP_VALUE_MAX];
    if (__system_property_get("debug.llmhub.ggml_vk_env", spec) <= 0) {
        return;
    }
    std::string all(spec);
    for (size_t start = 0; start < all.size();) {
        size_t end = all.find(';', start);
        std::string item = all.substr(start, end == std::string::npos ? std::string::npos : end - start);
        size_t eq = item.find('=');
        if (eq != std::string::npos) {
            setenv(item.substr(0, eq).c_str(), item.substr(eq + 1).c_str(), 1);
            LOGI("debug env %s", item.c_str());
        }
        if (end == std::string::npos) break;
        start = end + 1;
    }
}

extern "C" JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void*) {
    g_vm = vm;
    apply_debug_ggml_env();
    sd_set_log_callback(on_sd_log, nullptr);
    // ggml prints abort reasons to stderr, which Android discards; keep them in logcat and tombstones.
    ggml_set_abort_callback([](const char* message) {
        __android_log_print(ANDROID_LOG_FATAL, LOG_TAG, "ggml abort: %s", message);
        android_set_abort_message(message);
    });
    sd_set_progress_callback(on_sd_progress, nullptr);
    return JNI_VERSION_1_6;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_llmhub_llmhub_videogen_SdCppNative_nativeListDevices(JNIEnv* env, jobject) {
    std::vector<char> buffer(4096, 0);
    sd_list_devices(buffer.data(), buffer.size());
    return env->NewStringUTF(buffer.data());
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_llmhub_llmhub_videogen_SdCppNative_nativeLastError(JNIEnv* env, jobject) {
    return env->NewStringUTF(g_last_error.c_str());
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_llmhub_llmhub_videogen_SdCppNative_nativeCreate(JNIEnv* env,
                                                         jobject,
                                                         jstring diffusion_path,
                                                         jstring t5_path,
                                                         jstring vae_path,
                                                         jstring backend,
                                                         jint threads,
                                                         jboolean flash_attn,
                                                         jstring model_args) {
    g_last_error.clear();
    std::string diffusion = to_string(env, diffusion_path);
    std::string t5        = to_string(env, t5_path);
    std::string vae       = to_string(env, vae_path);
    std::string backend_s = to_string(env, backend);
    std::string args_s    = to_string(env, model_args);

    sd_ctx_params_t params;
    sd_ctx_params_init(&params);
    params.diffusion_model_path = diffusion.c_str();
    params.t5xxl_path           = t5.c_str();
    params.vae_path             = vae.c_str();
    params.n_threads            = threads > 0 ? threads : sd_get_num_physical_cores();
    params.enable_mmap          = true;
    params.flash_attn           = flash_attn;
    params.diffusion_flash_attn = flash_attn;
    params.backend              = backend_s.empty() ? nullptr : backend_s.c_str();
    // Text encoder weights are only needed for one pass per prompt: keep them on disk and stream
    // them in, so the T5 encoder and the DiT never have to be resident at the same time. On the NPU
    // the CPU VAE only encodes image-to-video start frames (decoding runs on the HTP), so its weights
    // are streamed the same way instead of holding 2.4 GB for the whole session.
    const bool npu        = args_s.find("ltxv_qnn_dir=") != std::string::npos;
    params.params_backend = npu ? "te=disk,vae=disk" : "te=disk";
    params.auto_fit       = true;
    // Diagnostics: `adb shell setprop debug.llmhub.flash_attn_mode off` (or conv_direct_mode) overrides
    // the per-backend automatic choice, to find GPU drivers that compute an optimized kernel wrongly.
    static char fa_mode[PROP_VALUE_MAX], conv_mode[PROP_VALUE_MAX];
    if (__system_property_get("debug.llmhub.flash_attn_mode", fa_mode) > 0) {
        params.flash_attn_mode = fa_mode;
    }
    if (__system_property_get("debug.llmhub.conv_direct_mode", conv_mode) > 0) {
        params.conv_direct_mode = conv_mode;
    }
    params.model_args     = args_s.empty() ? nullptr : args_s.c_str();

    LOGI("creating sd context: backend=%s threads=%d flash_attn=%d", backend_s.c_str(), params.n_threads, (int)flash_attn);
    sd_ctx_t* ctx = new_sd_ctx(&params);
    if (ctx == nullptr) {
        if (g_last_error.empty()) {
            g_last_error = "failed to create stable-diffusion context";
        }
        return 0;
    }
    if (!sd_ctx_supports_video_generation(ctx)) {
        g_last_error = "model does not support video generation";
        free_sd_ctx(ctx);
        return 0;
    }
    LOGI("loaded %s", sd_get_model_version_name(ctx));
    auto* session = new Session();
    session->ctx  = ctx;
    return reinterpret_cast<jlong>(session);
}

extern "C" JNIEXPORT jobjectArray JNICALL
Java_com_llmhub_llmhub_videogen_SdCppNative_nativeGenerate(JNIEnv* env,
                                                           jobject,
                                                           jlong handle,
                                                           jstring prompt,
                                                           jstring negative_prompt,
                                                           jint width,
                                                           jint height,
                                                           jint frames,
                                                           jint fps,
                                                           jint steps,
                                                           jfloat cfg_scale,
                                                           jlong seed,
                                                           jfloatArray sigmas,
                                                           jbyteArray init_rgb,
                                                           jint init_width,
                                                           jint init_height,
                                                           jobject callback) {
    auto* session = reinterpret_cast<Session*>(handle);
    if (session == nullptr || session->ctx == nullptr) {
        g_last_error = "video model is not loaded";
        return nullptr;
    }
    g_last_error.clear();
    session->cancelled = false;
    sd_cancel_generation(session->ctx, SD_CANCEL_RESET);

    {
        std::lock_guard<std::mutex> lock(g_callback_mutex);
        jclass cls            = env->GetObjectClass(callback);
        g_target.callback     = env->NewGlobalRef(callback);
        g_target.on_progress  = env->GetMethodID(cls, "onProgress", "(II)V");
        g_target.on_stage     = env->GetMethodID(cls, "onStage", "(I)V");
        env->DeleteLocalRef(cls);
    }

    std::string prompt_s   = to_string(env, prompt);
    std::string negative_s = to_string(env, negative_prompt);

    std::vector<float> sigma_values;
    if (sigmas != nullptr) {
        jsize count = env->GetArrayLength(sigmas);
        sigma_values.resize(static_cast<size_t>(count));
        env->GetFloatArrayRegion(sigmas, 0, count, sigma_values.data());
    }

    std::vector<uint8_t> init_pixels;
    if (init_rgb != nullptr) {
        jsize count = env->GetArrayLength(init_rgb);
        init_pixels.resize(static_cast<size_t>(count));
        env->GetByteArrayRegion(init_rgb, 0, count, reinterpret_cast<jbyte*>(init_pixels.data()));
    }

    sd_vid_gen_params_t params;
    sd_vid_gen_params_init(&params);
    params.prompt                                = prompt_s.c_str();
    params.negative_prompt                       = negative_s.c_str();
    params.width                                 = width;
    params.height                                = height;
    params.video_frames                          = frames;
    params.fps                                   = fps;
    params.seed                                  = seed;
    params.strength                              = 1.0f;
    params.sample_params.sample_steps            = steps;
    params.sample_params.sample_method           = EULER_SAMPLE_METHOD;
    params.sample_params.guidance.txt_cfg        = cfg_scale;
    if (!sigma_values.empty()) {
        params.sample_params.custom_sigmas       = sigma_values.data();
        params.sample_params.custom_sigmas_count = static_cast<int>(sigma_values.size());
    }
    if (!init_pixels.empty() && init_width > 0 && init_height > 0 &&
        init_pixels.size() == static_cast<size_t>(init_width) * static_cast<size_t>(init_height) * 3) {
        params.init_image = {static_cast<uint32_t>(init_width),
                             static_cast<uint32_t>(init_height),
                             3,
                             init_pixels.data()};
    }
    // Decode the video in short temporal tiles so the VAE activations fit in phone memory. Spatial
    // tiles alone still decode all frames at once, which got the app killed by the low-memory killer
    // at "Decoding frames" on CPU (e.g. Snapdragon 7s Gen 4); sd.cpp only adds temporal tiles after a
    // failed allocation, and Android kills the process before an allocation fails. The NPU decoder
    // ignores these settings (one compiled graph for the whole clip).
    params.vae_tiling_params.enabled         = true;
    params.vae_tiling_params.temporal_tiling = true;

    g_expected_steps = sigma_values.empty() ? steps : static_cast<int>(sigma_values.size()) - 1;
    emit_stage(STAGE_ENCODING_PROMPT);
    sd_image_t* out_frames = nullptr;
    int out_count          = 0;
    int out_fps            = fps;
    bool ok                = generate_video(session->ctx, &params, &out_frames, &out_count, nullptr, &out_fps);

    g_expected_steps = 0;
    {
        std::lock_guard<std::mutex> lock(g_callback_mutex);
        env->DeleteGlobalRef(g_target.callback);
        g_target = {};
    }

    if (!ok || out_frames == nullptr || out_count <= 0) {
        if (out_frames != nullptr) {
            free_sd_images(out_frames, out_count);
        }
        if (g_last_error.empty()) {
            g_last_error = session->cancelled ? "cancelled" : "video generation failed";
        }
        return nullptr;
    }

    // Frames come back as packed RGB; hand them to Kotlin as one byte[] per frame.
    jclass byte_array_cls = env->FindClass("[B");
    jobjectArray result   = env->NewObjectArray(out_count + 1, byte_array_cls, nullptr);
    // Element 0 carries the frame geometry: width, height, fps as little-endian int32.
    int32_t header[3] = {static_cast<int32_t>(out_frames[0].width), static_cast<int32_t>(out_frames[0].height), out_fps};
    jbyteArray header_array = env->NewByteArray(sizeof(header));
    env->SetByteArrayRegion(header_array, 0, sizeof(header), reinterpret_cast<const jbyte*>(header));
    env->SetObjectArrayElement(result, 0, header_array);
    env->DeleteLocalRef(header_array);
    for (int i = 0; i < out_count; i++) {
        const sd_image_t& frame = out_frames[i];
        jsize size              = static_cast<jsize>(frame.width * frame.height * frame.channel);
        jbyteArray array        = env->NewByteArray(size);
        if (array == nullptr) {
            break;
        }
        env->SetByteArrayRegion(array, 0, size, reinterpret_cast<const jbyte*>(frame.data));
        env->SetObjectArrayElement(result, i + 1, array);
        env->DeleteLocalRef(array);
    }
    free_sd_images(out_frames, out_count);
    return result;
}

extern "C" JNIEXPORT void JNICALL
Java_com_llmhub_llmhub_videogen_SdCppNative_nativeCancel(JNIEnv*, jobject, jlong handle) {
    auto* session = reinterpret_cast<Session*>(handle);
    if (session != nullptr && session->ctx != nullptr) {
        session->cancelled = true;
        sd_cancel_generation(session->ctx, SD_CANCEL_ALL);
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_llmhub_llmhub_videogen_SdCppNative_nativeFree(JNIEnv*, jobject, jlong handle) {
    auto* session = reinterpret_cast<Session*>(handle);
    if (session == nullptr) {
        return;
    }
    if (session->ctx != nullptr) {
        free_sd_ctx(session->ctx);
    }
    delete session;
}
