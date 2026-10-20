#include <android/log.h>
#include <jni.h>
#include <sys/stat.h>
#include <atomic>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include "stable-diffusion.h"
#include "json.hpp"

namespace {
using json = nlohmann::json;
struct ImageSession {
    sd_ctx_t* ctx;
    bool tile_vae = false;
    std::mutex generation;
    explicit ImageSession(sd_ctx_t* value) : ctx(value) {}
    ~ImageSession() { free_sd_ctx(ctx); }
};
std::mutex sessions_mutex;
std::unordered_map<jlong, std::shared_ptr<ImageSession>> sessions;
std::atomic<jlong> next_id{1};
std::shared_ptr<ImageSession> session(jlong id) {
    std::lock_guard<std::mutex> lock(sessions_mutex);
    auto it = sessions.find(id);
    return it == sessions.end() ? nullptr : it->second;
}
std::string text(JNIEnv* env, jstring value) {
    if (!value) return {};
    const char* chars = env->GetStringUTFChars(value, nullptr);
    if (!chars) return {};
    std::string result(chars);
    env->ReleaseStringUTFChars(value, chars);
    return result;
}
void fail(JNIEnv* env, const char* message) {
    if (!env->ExceptionCheck()) env->ThrowNew(env->FindClass("java/lang/IllegalStateException"), message);
}
int64_t file_bytes(const std::string& path) {
    struct stat info {};
    return path.empty() || stat(path.c_str(), &info) != 0 ? 0 : static_cast<int64_t>(info.st_size);
}
int64_t meminfo_bytes(const char* field) {
    std::ifstream meminfo("/proc/meminfo");
    std::string key, unit;
    int64_t kib = 0;
    while (meminfo >> key >> kib >> unit) {
        if (key == field) return kib * 1024;
    }
    return -1;
}
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_llmhub_llmhub_imagegen_SdCppImageNative_create(JNIEnv* env, jobject, jstring checkpoint, jstring options_json, jboolean gpu) {
    try {
        auto path = text(env, checkpoint);
        auto options = json::parse(text(env, options_json));
        std::string clip_l, clip_g, encoder, vae;
        std::vector<std::string> names, paths;
        for (const auto& asset : options["assets"]) {
            if (!asset.value("enabled", true)) continue;
            auto kind = asset.value("kind", "");
            auto file = asset.value("path", "");
            if (kind == "clipL") clip_l = file;
            else if (kind == "clipG") clip_g = file;
            else if (kind == "textEncoder") encoder = file;
            else if (kind == "vae") vae = file;
            else if (kind == "embedding") {
                names.push_back(asset.value("name", "")); paths.push_back(file);
            }
        }
        std::vector<sd_embedding_t> embeddings;
        for (size_t i = 0; i < names.size(); ++i) embeddings.push_back({names[i].c_str(), paths[i].c_str()});
        sd_ctx_params_t params;
        sd_ctx_params_init(&params);
        params.model_path = path.c_str();
        params.clip_l_path = clip_l.empty() ? nullptr : clip_l.c_str();
        params.clip_g_path = clip_g.empty() ? nullptr : clip_g.c_str();
        params.t5xxl_path = encoder.empty() ? nullptr : encoder.c_str();
        params.vae_path = vae.empty() ? nullptr : vae.c_str();
        params.embeddings = embeddings.data();
        params.embedding_count = static_cast<uint32_t>(embeddings.size());
        params.enable_mmap = true;
        params.auto_fit = true;
        params.n_threads = sd_get_num_physical_cores();
        // CPU text/VAE avoids mobile Vulkan failures with embeddings and image input.
        params.backend = gpu ? "te=CPU,diffusion=Vulkan0,vae=CPU" : "CPU";
        // Phone GPUs share RAM with the CPU, but Adreno reports a Vulkan heap larger than physical memory, so the
        // planner can place full-precision SDXL weights in memory that does not exist. Quantize float checkpoints to
        // Q8_0 at load when their weights would not leave room for compute buffers. GGUF files are already sized.
        const int64_t weights = file_bytes(path) + file_bytes(clip_l) + file_bytes(clip_g) + file_bytes(encoder) + file_bytes(vae);
        const int64_t available = meminfo_bytes("MemAvailable:");
        const bool float_checkpoint = path.size() > 12 && path.compare(path.size() - 12, 12, ".safetensors") == 0;
        if (float_checkpoint && available > 0 && weights > available - (3LL << 30)) {
            params.wtype = SD_TYPE_Q8_0;
            // AUTO picks the LoRA mode from the file's float types; merging into converted Q8_0 weights aborts.
            params.lora_apply_mode = LORA_APPLY_AT_RUNTIME;
            __android_log_print(ANDROID_LOG_INFO, "SdCppJni", "image weights %lld MiB exceed available %lld MiB less reserve; loading as Q8_0",
                                static_cast<long long>(weights >> 20), static_cast<long long>(available >> 20));
        }
        auto* ctx = new_sd_ctx(&params);
        if (!ctx) { fail(env, "Checkpoint or encoder loading failed"); return 0; }
        auto value = std::make_shared<ImageSession>(ctx);
        // 8 GB phones are killed decoding a whole SDXL image after diffusion; tiles bound VAE memory.
        const int64_t total = meminfo_bytes("MemTotal:");
        value->tile_vae = total > 0 && total < (10LL << 30);
        if (!sd_ctx_supports_image_generation(ctx)) { fail(env, "Checkpoint does not support image generation"); return 0; }
        jlong id = next_id++;
        std::lock_guard<std::mutex> lock(sessions_mutex);
        sessions.emplace(id, std::move(value));
        return id;
    } catch (const std::exception& e) { fail(env, e.what()); return 0; }
}

extern "C" JNIEXPORT jobjectArray JNICALL
Java_com_llmhub_llmhub_imagegen_SdCppImageNative_generate(JNIEnv* env, jobject, jlong id, jstring positive, jstring options_json,
                                                       jint width, jint height, jint steps, jlong seed, jfloat strength, jbyteArray input) {
    try {
        auto value = session(id);
        if (!value || width < 64 || height < 64 || width > 2048 || height > 2048 || width % 64 || height % 64) {
            fail(env, "Invalid image generation request"); return nullptr;
        }
        std::lock_guard<std::mutex> lock(value->generation);
        auto options = json::parse(text(env, options_json));
        auto prompt = text(env, positive);
        auto negative = options.value("negativePrompt", "");
        std::vector<std::string> files;
        std::vector<float> weights;
        for (const auto& asset : options["assets"]) {
            if (asset.value("enabled", true) && asset.value("kind", "") == "lora") {
                files.push_back(asset.value("path", "")); weights.push_back(asset.value("weight", 1.f));
            }
        }
        std::vector<sd_lora_t> loras;
        for (size_t i = 0; i < files.size(); ++i) loras.push_back({false, weights[i], files[i].c_str()});
        std::vector<uint8_t> rgb;
        if (input) {
            const auto bytes = env->GetArrayLength(input);
            if (bytes != width * height * 3) { fail(env, "Invalid conditioning image size"); return nullptr; }
            rgb.resize(bytes);
            env->GetByteArrayRegion(input, 0, bytes, reinterpret_cast<jbyte*>(rgb.data()));
        }
        sd_img_gen_params_t params;
        sd_img_gen_params_init(&params);
        params.prompt = prompt.c_str(); params.negative_prompt = negative.c_str();
        params.width = width; params.height = height; params.seed = seed; params.batch_count = 1;
        params.clip_skip = options.value("clipSkip", 1);
        params.loras = loras.data(); params.lora_count = static_cast<uint32_t>(loras.size());
        params.sample_params.sample_steps = steps;
        params.sample_params.sample_method = EULER_SAMPLE_METHOD;
        params.sample_params.guidance.txt_cfg = options.value("guidance", 7.f);
        params.strength = rgb.empty() ? 1.f : strength;
        params.vae_tiling_params.enabled = value->tile_vae;
        if (!rgb.empty()) params.init_image = {static_cast<uint32_t>(width), static_cast<uint32_t>(height), 3, rgb.data()};
        sd_cancel_generation(value->ctx, SD_CANCEL_RESET);
        sd_image_t* images = nullptr;
        int count = 0;
        bool generated = generate_image(value->ctx, &params, &images, &count);
        struct ImagesGuard { sd_image_t* data; int count; ~ImagesGuard() { if (data) free_sd_images(data, count); } } guard{images, count};
        if (!generated || count < 1 || !images[0].data || images[0].channel != 3) {
            fail(env, "Image generation failed"); return nullptr;
        }
        uint32_t dimensions[] = {images[0].width, images[0].height};
        auto header = env->NewByteArray(sizeof(dimensions));
        env->SetByteArrayRegion(header, 0, sizeof(dimensions), reinterpret_cast<jbyte*>(dimensions));
        auto size = images[0].width * images[0].height * 3;
        auto pixels = env->NewByteArray(size);
        env->SetByteArrayRegion(pixels, 0, size, reinterpret_cast<jbyte*>(images[0].data));
        auto result = env->NewObjectArray(2, env->FindClass("[B"), nullptr);
        env->SetObjectArrayElement(result, 0, header); env->SetObjectArrayElement(result, 1, pixels);
        return result;
    } catch (const std::exception& e) { fail(env, e.what()); return nullptr; }
}

extern "C" JNIEXPORT void JNICALL
Java_com_llmhub_llmhub_imagegen_SdCppImageNative_close(JNIEnv*, jobject, jlong id) {
    auto value = session(id);
    if (!value) return;
    sd_cancel_generation(value->ctx, SD_CANCEL_ALL);
    std::lock_guard<std::mutex> lock(sessions_mutex);
    sessions.erase(id);
}
