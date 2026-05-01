// Mimo bot — JNI bridge from Kotlin to espeak-ng for text→IPA phonemization.
//
// Symbols exported here are loaded by EspeakG2P.kt via System.loadLibrary.
// All entry points are namespace `Java_com_llmhub_llmhub_mimobot_speech_kokoro_EspeakG2P_*`.
//
// espeak-ng API reference: speak_lib.h
//   espeak_Initialize(AUDIO_OUTPUT_SYNCHRONOUS, 0, dataPath, 0)
//   espeak_SetVoiceByName("en-us")
//   espeak_TextToPhonemes(&inputBuf, espeakCHARS_AUTO,
//                         espeakPHONEMES_IPA | espeakPHONEMES_SHOW)
//
// We deliberately keep this thin: tokenisation / vocabulary mapping happens in
// Kotlin (KokoroVocab) so we can swap espeak for another phonemizer (e.g.
// misaki, custom CMU dict) without touching native code.

#include <jni.h>
#include <cstring>
#include <string>
#include <android/log.h>

extern "C" {
#include "speak_lib.h"
}

#define LOG_TAG "MimoEspeakJni"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)

static bool g_initialised = false;

extern "C" JNIEXPORT jint JNICALL
Java_com_llmhub_llmhub_mimobot_speech_kokoro_EspeakG2P_nativeInit(
    JNIEnv* env, jclass /*clazz*/, jstring jDataPath) {
    if (g_initialised) return 0;
    const char* dataPath = env->GetStringUTFChars(jDataPath, nullptr);
    // AUDIO_OUTPUT_SYNCHRONOUS with no buffer — we never play audio, only ask
    // for phonemes. Returning sample rate or -1 on failure.
    int rate = espeak_Initialize(AUDIO_OUTPUT_SYNCHRONOUS, 0, dataPath, 0);
    env->ReleaseStringUTFChars(jDataPath, dataPath);
    if (rate < 0) {
        LOGW("espeak_Initialize failed (rate=%d)", rate);
        return rate;
    }
    g_initialised = true;
    LOGI("espeak-ng initialised (rate=%d)", rate);
    return 0;
}

extern "C" JNIEXPORT jint JNICALL
Java_com_llmhub_llmhub_mimobot_speech_kokoro_EspeakG2P_nativeSetVoice(
    JNIEnv* env, jclass /*clazz*/, jstring jVoiceName) {
    const char* voice = env->GetStringUTFChars(jVoiceName, nullptr);
    espeak_ERROR rc = espeak_SetVoiceByName(voice);
    env->ReleaseStringUTFChars(jVoiceName, voice);
    if (rc != EE_OK) {
        LOGW("espeak_SetVoiceByName(%s) failed rc=%d", voice, (int)rc);
        return -1;
    }
    return 0;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_llmhub_llmhub_mimobot_speech_kokoro_EspeakG2P_nativePhonemize(
    JNIEnv* env, jclass /*clazz*/, jstring jText) {
    if (!g_initialised) return env->NewStringUTF("");
    const char* utf = env->GetStringUTFChars(jText, nullptr);

    // espeak_TextToPhonemes:
    //   - mode bits: 0x02 = IPA, 0x01 = print stress markers (we want both).
    //     espeakPHONEMES_IPA  = 0x02
    //     espeakPHONEMES_SHOW = 0x01 (returned, not printed when no FILE*)
    //   - input is consumed; we pass &input which the function advances.
    const void* input = utf;
    std::string out;
    while (input != nullptr) {
        const char* phonemes = espeak_TextToPhonemes(
            &input,
            /*textmode*/ espeakCHARS_AUTO,
            /*phonememode*/ 0x02);  // IPA, no stress separators
        if (phonemes == nullptr) break;
        if (!out.empty() && out.back() != ' ') out.push_back(' ');
        out.append(phonemes);
    }

    env->ReleaseStringUTFChars(jText, utf);
    return env->NewStringUTF(out.c_str());
}

extern "C" JNIEXPORT void JNICALL
Java_com_llmhub_llmhub_mimobot_speech_kokoro_EspeakG2P_nativeShutdown(
    JNIEnv* /*env*/, jclass /*clazz*/) {
    if (!g_initialised) return;
    espeak_Terminate();
    g_initialised = false;
}
