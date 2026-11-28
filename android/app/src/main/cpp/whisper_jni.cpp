// whisper.cpp speech recognition. Built twice from this file:
//  - libllmhub_whisper (WhisperCppNative): against the ggml of the vendored llama.cpp, so it runs on the same
//    per-chip CPU backend (libllmhub_ggml-cpu-*) as the GGUF chat models, or on its Vulkan module.
//  - libllmhub_whisper_snapdragon (WhisperCppSnapdragonNative, LLMHUB_WHISPER_SNAPDRAGON): against the official
//    llama.cpp Snapdragon release in jniLibs, for the Adreno GPU (OpenCL) and the Hexagon NPU.

#include <android/log.h>
#include <dirent.h>
#include <dlfcn.h>
#include <jni.h>

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#include "ggml-backend.h"
#include "ggml.h"
#include "whisper.h"

#ifdef LLMHUB_WHISPER_SNAPDRAGON
#define WHISPER_JNI(name) Java_com_llmhub_llmhub_inference_WhisperCppSnapdragonNative_##name
#else
#define WHISPER_JNI(name) Java_com_llmhub_llmhub_inference_WhisperCppNative_##name
#endif

namespace {

constexpr const char * TAG = "WhisperCpp";

std::mutex g_init_mutex;
bool g_backend_ready = false;
std::atomic<bool> g_abort{false};
// The accelerator's backend, initialized before whisper.cpp loads the model (see nativeLoad) and kept until the
// model is freed. One model is loaded at a time.
ggml_backend_t g_device_backend = nullptr;

std::string from_jstring(JNIEnv * env, jstring value) {
    if (value == nullptr) return {};
    const char * chars = env->GetStringUTFChars(value, nullptr);
    std::string result = chars != nullptr ? chars : "";
    if (chars != nullptr) env->ReleaseStringUTFChars(value, chars);
    return result;
}

void log_callback(ggml_log_level level, const char * text, void *) {
    if (level >= GGML_LOG_LEVEL_WARN) {
        __android_log_write(level == GGML_LOG_LEVEL_ERROR ? ANDROID_LOG_ERROR : ANDROID_LOG_WARN, TAG, text);
    }
}

// GGML_ASSERT/GGML_ABORT print to stderr, which Android drops; put the message in logcat before the abort.
void abort_callback(const char * message) {
    __android_log_print(ANDROID_LOG_FATAL, TAG, "ggml abort: %s", message != nullptr ? message : "");
}

#ifndef LLMHUB_WHISPER_SNAPDRAGON
// The same choice as llama_cpp_jni.cpp's load_cpu_backend: the libllmhub_ggml-cpu-<variant>.so with the highest
// ggml_backend_score() on this CPU. Both libraries share one ggml, so whichever runs first registers it.
bool load_cpu_backend(const std::string & library_dir) {
    if (ggml_backend_reg_by_name("CPU") != nullptr) return true;
    constexpr const char * prefix = "libllmhub_ggml-cpu-";
    constexpr const char * suffix = ".so";
    DIR * dir = opendir(library_dir.c_str());
    if (dir == nullptr) {
        __android_log_print(ANDROID_LOG_ERROR, TAG, "Cannot list native libraries in %s", library_dir.c_str());
        return false;
    }
    std::string best_path;
    int best_score = 0;
    while (const dirent * entry = readdir(dir)) {
        const std::string name = entry->d_name;
        if (name.rfind(prefix, 0) != 0 || name.size() <= std::strlen(suffix) ||
            name.compare(name.size() - std::strlen(suffix), std::string::npos, suffix) != 0) {
            continue;
        }
        const std::string path = library_dir + "/" + name;
        void * handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (handle == nullptr) continue;
        auto score_fn = reinterpret_cast<int (*)()>(dlsym(handle, "ggml_backend_score"));
        const int score = score_fn != nullptr ? score_fn() : 0;
        dlclose(handle);
        if (score > best_score) {
            best_score = score;
            best_path = path;
        }
    }
    closedir(dir);
    if (best_path.empty() || ggml_backend_load(best_path.c_str()) == nullptr) {
        __android_log_print(ANDROID_LOG_ERROR, TAG, "No CPU backend variant loaded from %s", library_dir.c_str());
        return false;
    }
    __android_log_print(ANDROID_LOG_INFO, TAG, "CPU backend: %s (score %d)",
                        best_path.substr(best_path.rfind('/') + 1).c_str(), best_score);
    return true;
}
#endif

bool load_backend(const char * registry_name, const std::string & path) {
    if (ggml_backend_reg_by_name(registry_name) != nullptr) return true;
    if (ggml_backend_load(path.c_str()) != nullptr) return true;
    __android_log_print(ANDROID_LOG_ERROR, TAG, "Cannot load %s", path.c_str());
    return false;
}

/** Index of the GPU-type device named [name] among GPU/IGPU devices (whisper_context_params.gpu_device), or -1. */
int gpu_index(const std::string & name) {
    int index = 0;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const auto type = ggml_backend_dev_type(dev);
        if (type != GGML_BACKEND_DEVICE_TYPE_GPU && type != GGML_BACKEND_DEVICE_TYPE_IGPU) continue;
        if (name == ggml_backend_dev_name(dev)) return index;
        ++index;
    }
    return -1;
}

} // namespace

// Registers the CPU backend and, when [accelerator] is set, its module: "Vulkan" (source build), "OpenCL" or
// "Hexagon" (Snapdragon build; [htp_dir] holds the Hexagon skels). Only the requested module is loaded, so a CPU
// or GPU load never probes another driver.
extern "C" JNIEXPORT jboolean JNICALL
WHISPER_JNI(nativeInit)(JNIEnv * env, jobject, jstring library_dir, jstring accelerator, jstring htp_dir) {
    std::lock_guard<std::mutex> lock(g_init_mutex);
    const std::string dir = from_jstring(env, library_dir);
    const std::string wanted = from_jstring(env, accelerator);
    if (!g_backend_ready) {
        whisper_log_set(log_callback, nullptr);
        ggml_set_abort_callback(abort_callback);
#ifdef LLMHUB_WHISPER_SNAPDRAGON
        g_backend_ready = load_backend("CPU", dir + "/libggml-cpu.so");
#else
        g_backend_ready = load_cpu_backend(dir);
#endif
    }
    if (!g_backend_ready) return JNI_FALSE;
#ifdef LLMHUB_WHISPER_SNAPDRAGON
    if (wanted == "OpenCL") return load_backend("OpenCL", dir + "/libggml-opencl.so") ? JNI_TRUE : JNI_FALSE;
    if (wanted == "Hexagon") {
        const std::string htp = from_jstring(env, htp_dir);
        if (!htp.empty()) {
            const char * old_path = std::getenv("ADSP_LIBRARY_PATH");
            if (old_path == nullptr || std::strstr(old_path, htp.c_str()) == nullptr) {
                const std::string path = htp + (old_path != nullptr ? ";" + std::string(old_path) : "");
                setenv("ADSP_LIBRARY_PATH", path.c_str(), 1);
            }
        }
        return load_backend("HTP", dir + "/libggml-hexagon.so") ? JNI_TRUE : JNI_FALSE;
    }
#else
    (void) htp_dir;
    if (wanted == "Vulkan") return load_backend("Vulkan", dir + "/libllmhub_ggml-vulkan.so") ? JNI_TRUE : JNI_FALSE;
#endif
    return wanted.empty() ? JNI_TRUE : JNI_FALSE;
}

/** Names of the GPU-type devices whisper.cpp can use, for picking [nativeLoad]'s device. */
extern "C" JNIEXPORT jstring JNICALL
WHISPER_JNI(nativeGpuDevices)(JNIEnv * env, jobject) {
    std::string names;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const auto type = ggml_backend_dev_type(dev);
        if (type != GGML_BACKEND_DEVICE_TYPE_GPU && type != GGML_BACKEND_DEVICE_TYPE_IGPU) continue;
        if (!names.empty()) names += ',';
        names += ggml_backend_dev_name(dev);
    }
    return env->NewStringUTF(names.c_str());
}

// [device] is a ggml device name (e.g. "Vulkan0", "GPUOpenCL", "HTP0") to run the model on, or null for the CPU.
extern "C" JNIEXPORT jlong JNICALL
WHISPER_JNI(nativeLoad)(JNIEnv * env, jobject, jstring model_path, jstring device, jboolean flash_attn) {
    whisper_context_params params = whisper_context_default_params();
    const std::string device_name = from_jstring(env, device);
    params.use_gpu = !device_name.empty();
    if (params.use_gpu) {
        params.gpu_device = gpu_index(device_name);
        if (params.gpu_device < 0) {
            __android_log_print(ANDROID_LOG_ERROR, TAG, "No device %s", device_name.c_str());
            return 0;
        }
    }
    params.flash_attn = flash_attn == JNI_TRUE;
    if (g_device_backend != nullptr) {
        ggml_backend_free(g_device_backend);
        g_device_backend = nullptr;
    }
    if (params.use_gpu) {
        // whisper.cpp allocates the weights in the device's buffers before it creates the device's backend. The
        // OpenCL backend reads its context (alignment, max size) from a backend that does not exist yet and
        // crashes, so initialize the device first, as llama.cpp's load order does.
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (device_name == ggml_backend_dev_name(dev)) {
                g_device_backend = ggml_backend_dev_init(dev, nullptr);
                break;
            }
        }
        if (g_device_backend == nullptr) {
            __android_log_print(ANDROID_LOG_ERROR, TAG, "Cannot initialize %s", device_name.c_str());
            return 0;
        }
    }
    whisper_context * ctx = whisper_init_from_file_with_params(from_jstring(env, model_path).c_str(), params);
    if (ctx == nullptr) __android_log_write(ANDROID_LOG_ERROR, TAG, "whisper_init_from_file_with_params failed");
    return reinterpret_cast<jlong>(ctx);
}

// Transcribes 16 kHz mono float samples. Returns the text as UTF-8 bytes (whisper.cpp can split a multi-byte
// character across segments, which NewStringUTF would reject), or null on failure.
extern "C" JNIEXPORT jbyteArray JNICALL
WHISPER_JNI(nativeTranscribe)(JNIEnv * env, jobject, jlong handle, jfloatArray samples, jstring language, jint threads) {
    auto * ctx = reinterpret_cast<whisper_context *>(handle);
    if (ctx == nullptr || samples == nullptr) return nullptr;
    const jsize count = env->GetArrayLength(samples);
    std::vector<float> pcm(static_cast<size_t>(count));
    env->GetFloatArrayRegion(samples, 0, count, pcm.data());
    const std::string lang = from_jstring(env, language);

    whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    params.n_threads = threads > 0 ? threads : 4;
    params.language = lang.empty() ? "auto" : lang.c_str();
    params.translate = false;
    params.no_timestamps = true;
    params.no_context = true;
    params.print_special = false;
    params.print_progress = false;
    params.print_realtime = false;
    params.print_timestamps = false;
    params.suppress_blank = true;
    params.suppress_nst = true;
    g_abort.store(false);
    params.abort_callback = [](void *) { return g_abort.load(); };

    if (whisper_full(ctx, params, pcm.data(), static_cast<int>(pcm.size())) != 0) {
        if (!g_abort.load()) __android_log_write(ANDROID_LOG_ERROR, TAG, "whisper_full failed");
        return nullptr;
    }
    std::string text;
    const int segments = whisper_full_n_segments(ctx);
    for (int i = 0; i < segments; ++i) {
        const char * piece = whisper_full_get_segment_text(ctx, i);
        if (piece != nullptr) text += piece;
    }
    jbyteArray result = env->NewByteArray(static_cast<jsize>(text.size()));
    env->SetByteArrayRegion(result, 0, static_cast<jsize>(text.size()), reinterpret_cast<const jbyte *>(text.data()));
    return result;
}

extern "C" JNIEXPORT void JNICALL
WHISPER_JNI(nativeAbort)(JNIEnv *, jobject) {
    g_abort.store(true);
}

extern "C" JNIEXPORT void JNICALL
WHISPER_JNI(nativeFree)(JNIEnv *, jobject, jlong handle) {
    whisper_free(reinterpret_cast<whisper_context *>(handle));
    if (g_device_backend != nullptr) {
        ggml_backend_free(g_device_backend);
        g_device_backend = nullptr;
    }
}
