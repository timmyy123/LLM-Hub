package com.llmhub.llmhub.mimobot.speech.kokoro

/**
 * Pluggable grapheme-to-phoneme converter. Implementations turn an utterance
 * of text into a space-joined IPA phoneme string using only symbols present in
 * [KokoroVocab.SYMBOLS].
 *
 * v0 ships two implementations:
 *   - [DictionaryG2P]  — bundled ~150-word English dictionary, letter-fallback
 *                        for OOV. Always available.
 *   - [EspeakG2P]      — espeak-ng via JNI. Multilingual, ~95%+ coverage,
 *                        proper letter-to-sound rules. Requires native
 *                        binaries (see scripts/build_espeak_android.sh and
 *                        docs/espeak-ng-setup.md).
 *
 * The Kokoro pipeline picks the best one available at startup via [G2P.best].
 */
interface G2P {
    /** Human-readable name shown in the test screen ("dictionary", "espeak-ng"). */
    val displayName: String

    /** Convert text → space-joined IPA. May return "" for empty / unsupported input. */
    fun phonemize(text: String, language: String = "en-us"): String

    companion object {
        /**
         * Return the best available G2P, attempting espeak-ng first and falling
         * back to the bundled dictionary when its native libs aren't present.
         * Pass an Android Context so we can stage espeak-ng's data files.
         */
        fun best(context: android.content.Context): G2P =
            EspeakG2P.tryLoad(context) ?: DictionaryG2P
    }
}
