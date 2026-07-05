#include <jni.h>
#include <string>
#include <vector>
#include <cstdio>
#include <cstdint>
#include <android/log.h>
#include "whisper.h"

#define TAG "WhisperJNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

extern "C" {

JNIEXPORT jlong JNICALL
Java_com_llmhub_llmhub_inference_WhisperCppService_nativeInitContext(
    JNIEnv *env, jobject /*thiz*/, jstring model_path) {

    const char *path = env->GetStringUTFChars(model_path, nullptr);
    struct whisper_context_params cparams = whisper_context_default_params();
    struct whisper_context *ctx = whisper_init_from_file_with_params(path, cparams);
    env->ReleaseStringUTFChars(model_path, path);

    if (!ctx) {
        LOGE("Failed to init whisper context");
        return 0;
    }
    LOGI("Whisper context initialized");
    return reinterpret_cast<jlong>(ctx);
}

JNIEXPORT void JNICALL
Java_com_llmhub_llmhub_inference_WhisperCppService_nativeFreeContext(
    JNIEnv * /*env*/, jobject /*thiz*/, jlong context_ptr) {

    auto *ctx = reinterpret_cast<struct whisper_context *>(context_ptr);
    if (ctx) {
        whisper_free(ctx);
        LOGI("Whisper context freed");
    }
}

JNIEXPORT jstring JNICALL
Java_com_llmhub_llmhub_inference_WhisperCppService_nativeTranscribe(
    JNIEnv *env, jobject /*thiz*/, jlong context_ptr, jstring wav_path, jstring language) {

    auto *ctx = reinterpret_cast<struct whisper_context *>(context_ptr);
    if (!ctx) {
        return env->NewStringUTF("");
    }

    const char *path = env->GetStringUTFChars(wav_path, nullptr);
    const char *lang = env->GetStringUTFChars(language, nullptr);

    // Read WAV file into PCM float samples
    std::vector<float> pcmf32;
    {
        FILE *f = fopen(path, "rb");
        if (!f) {
            LOGE("Cannot open WAV file: %s", path);
            env->ReleaseStringUTFChars(wav_path, path);
            env->ReleaseStringUTFChars(language, lang);
            return env->NewStringUTF("");
        }

        // Skip WAV header (44 bytes for standard PCM WAV)
        char header[44];
        size_t header_read = fread(header, 1, 44, f);
        if (header_read < 44) {
            fclose(f);
            env->ReleaseStringUTFChars(wav_path, path);
            env->ReleaseStringUTFChars(language, lang);
            return env->NewStringUTF("");
        }

        // Determine sample format from WAV header
        int16_t audio_format = *(int16_t *)(header + 20);
        int16_t bits_per_sample = *(int16_t *)(header + 34);
        int32_t sample_rate = *(int32_t *)(header + 24);

        // Read raw audio data
        fseek(f, 0, SEEK_END);
        long file_size = ftell(f);
        long data_size = file_size - 44;
        fseek(f, 44, SEEK_SET);

        if (audio_format == 1 && bits_per_sample == 16) {
            // PCM 16-bit
            int n_samples = data_size / 2;
            std::vector<int16_t> pcm16(n_samples);
            fread(pcm16.data(), 2, n_samples, f);
            pcmf32.resize(n_samples);
            for (int i = 0; i < n_samples; i++) {
                pcmf32[i] = static_cast<float>(pcm16[i]) / 32768.0f;
            }
        } else if (audio_format == 3 && bits_per_sample == 32) {
            // IEEE float 32-bit
            int n_samples = data_size / 4;
            pcmf32.resize(n_samples);
            fread(pcmf32.data(), 4, n_samples, f);
        } else {
            LOGE("Unsupported WAV format: fmt=%d bits=%d", audio_format, bits_per_sample);
            fclose(f);
            env->ReleaseStringUTFChars(wav_path, path);
            env->ReleaseStringUTFChars(language, lang);
            return env->NewStringUTF("");
        }
        fclose(f);

        // Resample to 16kHz if needed
        if (sample_rate != 16000 && sample_rate > 0) {
            float ratio = 16000.0f / (float)sample_rate;
            int new_size = (int)(pcmf32.size() * ratio);
            std::vector<float> resampled(new_size);
            for (int i = 0; i < new_size; i++) {
                float src_idx = (float)i / ratio;
                int idx = (int)src_idx;
                if (idx >= (int)pcmf32.size() - 1) idx = pcmf32.size() - 2;
                float frac = src_idx - idx;
                resampled[i] = pcmf32[idx] * (1.0f - frac) + pcmf32[idx + 1] * frac;
            }
            pcmf32 = std::move(resampled);
        }
    }

    env->ReleaseStringUTFChars(wav_path, path);

    // Run whisper inference
    struct whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    params.print_realtime = false;
    params.print_progress = false;
    params.print_timestamps = false;
    params.print_special = false;
    params.single_segment = false;
    params.no_timestamps = true;
    params.n_threads = 4;

    if (lang[0] != '\0') {
        params.language = lang;
    } else {
        params.language = nullptr;
        params.detect_language = true;
    }

    int ret = whisper_full(ctx, params, pcmf32.data(), pcmf32.size());
    env->ReleaseStringUTFChars(language, lang);

    if (ret != 0) {
        LOGE("whisper_full failed with code %d", ret);
        return env->NewStringUTF("");
    }

    // Collect all segments
    std::string result;
    int n_segments = whisper_full_n_segments(ctx);
    for (int i = 0; i < n_segments; i++) {
        const char *text = whisper_full_get_segment_text(ctx, i);
        if (text) {
            result += text;
        }
    }

    return env->NewStringUTF(result.c_str());
}

} // extern "C"
