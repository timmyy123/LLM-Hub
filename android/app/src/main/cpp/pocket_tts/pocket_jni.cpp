#include <jni.h>
#define PTT_EMBEDDED
#define PTT_SHARED_LIB
#include "pocket_tts.cpp"

// Upstream RNG/profiler are process-global. Serialize every engine operation.
static std::mutex engine_mutex;
// UI cloning and speech services share the same sessions rather than duplicating RAM.
static std::unordered_map<std::string, pocket_tts::PocketTTS*> engines;
static std::unordered_map<pocket_tts::PocketTTS*, size_t> references;
static std::string utf8(JNIEnv* env, jstring input) {
    // Java supplies UTF-8 bytes through String.getBytes rather than modified UTF-8.
    auto cls = env->FindClass("java/lang/String");
    auto method = env->GetMethodID(cls, "getBytes", "(Ljava/lang/String;)[B");
    auto charset = env->NewStringUTF("UTF-8");
    auto bytes = (jbyteArray)env->CallObjectMethod(input, method, charset);
    std::string result(env->GetArrayLength(bytes), '\0');
    env->GetByteArrayRegion(bytes, 0, result.size(), reinterpret_cast<jbyte*>(result.data()));
    env->DeleteLocalRef(bytes); env->DeleteLocalRef(charset); env->DeleteLocalRef(cls);
    return result;
}
static void fail(JNIEnv* env, const std::exception& error) {
    env->ThrowNew(env->FindClass("java/lang/IllegalStateException"), error.what());
}
extern "C" JNIEXPORT jlong JNICALL
Java_com_llmhub_llmhub_ui_components_PocketTtsEngine_nativeCreate(JNIEnv* env, jobject, jstring models, jstring voices) {
    std::lock_guard<std::mutex> lock(engine_mutex);
    try {
        pocket_tts::Config config;
        config.models_dir = utf8(env, models); config.voices_dir = utf8(env, voices);
        config.tokenizer_path = config.models_dir + "/tokenizer.model";
        config.num_threads = 4;
        std::string key = config.models_dir + "\n" + config.voices_dir;
        auto found = engines.find(key);
        if (found != engines.end()) {
            ++references[found->second];
            return reinterpret_cast<jlong>(found->second);
        }
        auto engine = std::make_unique<pocket_tts::PocketTTS>(config);
        engines.emplace(key, engine.get());
        references.emplace(engine.get(), 1);
        return reinterpret_cast<jlong>(engine.release());
    } catch (const std::exception& e) { fail(env, e); return 0; }
}
extern "C" JNIEXPORT void JNICALL
Java_com_llmhub_llmhub_ui_components_PocketTtsEngine_nativeEncode(JNIEnv* env, jobject, jlong handle, jstring path) {
    std::lock_guard<std::mutex> lock(engine_mutex);
    try { reinterpret_cast<pocket_tts::PocketTTS*>(handle)->encode_voice(utf8(env, path)); }
    catch (const std::exception& e) { fail(env, e); }
}
extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_llmhub_llmhub_ui_components_PocketTtsEngine_nativeSynthesize(JNIEnv* env, jobject, jlong handle, jstring text, jstring voice, jobject probe) {
    std::lock_guard<std::mutex> lock(engine_mutex);
    try {
        auto method = env->GetMethodID(env->GetObjectClass(probe), "isCancelled", "()Z");
        if (env->CallBooleanMethod(probe, method)) return env->NewFloatArray(0);
        std::string reference = utf8(env, voice);
        if (!std::ifstream(reference).good()) throw std::runtime_error("Reference audio was deleted");
        std::vector<float> samples;
        reinterpret_cast<pocket_tts::PocketTTS*>(handle)->stream(utf8(env, text), reference, [&](const float* data, size_t size) {
            if (env->CallBooleanMethod(probe, method) || env->ExceptionCheck()) return false;
            if (samples.size() + size > 24000 * 60) throw std::runtime_error("Speech chunk too long");
            samples.insert(samples.end(), data, data + size);
            return true;
        }, 500);
        auto result = env->NewFloatArray(samples.size());
        if (result) env->SetFloatArrayRegion(result, 0, samples.size(), samples.data());
        return result;
    } catch (const std::exception& e) { fail(env, e); return nullptr; }
}
extern "C" JNIEXPORT void JNICALL
Java_com_llmhub_llmhub_ui_components_PocketTtsEngine_nativeClose(JNIEnv*, jobject, jlong handle) {
    std::lock_guard<std::mutex> lock(engine_mutex);
    auto engine = reinterpret_cast<pocket_tts::PocketTTS*>(handle);
    auto found = references.find(engine);
    if (found == references.end()) return;
    if (--found->second == 0) {
        for (auto it = engines.begin(); it != engines.end(); ++it) {
            if (it->second == engine) { engines.erase(it); break; }
        }
        references.erase(found);
        delete engine;
    }
}
