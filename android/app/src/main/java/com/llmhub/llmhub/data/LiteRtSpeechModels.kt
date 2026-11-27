package com.llmhub.llmhub.data

import android.content.Context
import android.os.Build
import java.io.File

/**
 * Google's `.litertlm` speech bundles (the formats AI Edge Gallery ships). Each is one file with the
 * voices packaged inside, run by the MediaPipe `tasks-audio` TextToSpeech / AudioTranscriber tasks.
 */
object LiteRtTtsModel {
    const val KOKORO_NAME = "Kokoro-82M (LiteRT-LM)"
    const val POCKET_NAME = "Pocket TTS (LiteRT-LM)"
    const val PHONON_NAME = "Phonon 31M (LiteRT-LM)"

    const val KOKORO_URL =
        "https://huggingface.co/litert-community/Kokoro-82M/resolve/81e43a025bccabcbbd487f08bc3ebd65b5fc7295/kokoro_int8_gpu_fp16.litertlm"
    const val KOKORO_SIZE = 294750530L
    const val POCKET_URL =
        "https://huggingface.co/litert-community/pocket-tts/resolve/bca93d6756489f9403f94d66826cfa65f80023a1/pocket_tts_int8.litertlm"
    const val POCKET_SIZE = 760061960L
    const val PHONON_URL =
        "https://huggingface.co/litert-community/phonon-31m-en/resolve/ee374c33f0f6f8ed1d9f761e21cdb97df6accfe8/phonon_int8.litertlm"
    const val PHONON_SIZE = 87205888L

    const val KOKORO_DEFAULT_VOICE = "af_heart"
    const val POCKET_DEFAULT_VOICE = "alba"
    const val PHONON_DEFAULT_VOICE = "freya"

    fun isModel(name: String?) = name == KOKORO_NAME || name == POCKET_NAME || name == PHONON_NAME
    fun isPocket(name: String?) = name == POCKET_NAME
    fun isPhonon(name: String?) = name == PHONON_NAME

    /**
     * Only the Kokoro bundle has a GPU build; PocketTTS is CPU only. Phonon loads on the GPU but took
     * 7.6 s to load there and was no faster than the CPU on the Tab S9+, so it stays on the CPU.
     * Kokoro also stays on the CPU on [kokoroGpuGarbledSocs].
     */
    fun supportsGpu(name: String?) = name == KOKORO_NAME && !kokoroGpuGarbled()

    /**
     * Chips whose GPU garbles Kokoro's vocoder (LiteRT OpenCL). On a Pixel 10 (Tensor G5) every Kokoro bundle
     * with the vocoder on the GPU turned speech into gibberish or near silence after a sentence or two, with
     * tasks-audio 1.2.0 and 1.2.3, while the CPU gave clean speech at the same speed (about 2 s a sentence).
     * Snapdragon 8 Gen 2, Exynos 1480 and Dimensity 9400 GPUs were clean.
     */
    private val kokoroGpuGarbledSocs = setOf("Tensor G5")

    private fun kokoroGpuGarbled() = Build.VERSION.SDK_INT >= Build.VERSION_CODES.S && Build.SOC_MODEL in kokoroGpuGarbledSocs

    fun file(context: Context, name: String): File {
        val model = ModelData.ttsModels.first { it.name == name }
        return File(File(context.filesDir, "models"), model.localFileName())
    }

    fun isComplete(context: Context, name: String): Boolean {
        val model = ModelData.ttsModels.firstOrNull { it.name == name } ?: return false
        val file = file(context, name)
        return file.isFile && file.length() >= (model.sizeBytes * 0.99).toLong()
    }

    private val kokoroLanguages = mapOf(
        'a' to ("American" to "en-US"), 'b' to ("British" to "en-GB"), 'e' to ("Spanish" to "es-ES"),
        'f' to ("French" to "fr-FR"), 'h' to ("Hindi" to "hi-IN"), 'i' to ("Italian" to "it-IT"),
        'j' to ("Japanese" to "ja-JP"), 'p' to ("Portuguese" to "pt-BR"), 'z' to ("Chinese" to "zh-CN")
    )

    private val kokoroVoiceIds = listOf(
        "af_heart", "af_alloy", "af_aoede", "af_bella", "af_jessica", "af_kore", "af_nicole", "af_nova",
        "af_river", "af_sarah", "af_sky", "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam",
        "am_michael", "am_onyx", "am_puck", "am_santa", "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
        "bm_daniel", "bm_fable", "bm_george", "bm_lewis", "ef_dora", "em_alex", "em_santa", "ff_siwis",
        "hf_alpha", "hf_beta", "hm_omega", "hm_psi", "if_sara", "im_nicola", "jf_alpha", "jf_gongitsune",
        "jf_nezumi", "jf_tebukuro", "jm_kumo", "pf_dora", "pm_alex", "pm_santa", "zf_xiaobei", "zf_xiaoni",
        "zf_xiaoxiao", "zf_xiaoyi", "zm_yunjian", "zm_yunxi", "zm_yunxia", "zm_yunyang"
    )

    /** Voice id to the locale its Flow-LM weights were trained for. */
    private val pocketVoiceLocales = linkedMapOf(
        "alba" to "en-US", "marius" to "en-US", "javert" to "en-US", "jean" to "en-US",
        "fantine" to "en-US", "cosette" to "en-US", "eponine" to "en-US", "azelma" to "en-US",
        "estelle" to "fr-FR", "juergen" to "de-DE", "lola" to "es-ES", "rafael" to "pt-BR",
        "giovanni" to "it-IT", "daan" to "nl-NL"
    )

    /** Phonon 31M's bundled voices, all American English. */
    private val phononVoiceIds = listOf("freya", "claire", "lukas", "sofia", "lucas")

    private val pocketLanguageNames = mapOf(
        "en-US" to "English", "fr-FR" to "French", "de-DE" to "German", "es-ES" to "Spanish",
        "pt-BR" to "Portuguese", "it-IT" to "Italian", "nl-NL" to "Dutch"
    )

    private fun String.capitalized() = replaceFirstChar { it.uppercase() }

    /** Voice id to display label, in the order the picker shows them. */
    fun voices(name: String?): List<Pair<String, String>> = when (name) {
        KOKORO_NAME -> kokoroVoiceIds.map { id ->
            val language = kokoroLanguages[id[0]]?.first ?: ""
            val gender = if (id[1] == 'f') "Female" else "Male"
            id to "$language $gender — ${id.substringAfter('_').capitalized()}"
        }
        POCKET_NAME -> pocketVoiceLocales.map { (id, locale) ->
            id to "${pocketLanguageNames[locale] ?: locale} — ${id.capitalized()}"
        }
        PHONON_NAME -> phononVoiceIds.map { id -> id to "English — ${id.capitalized()}" }
        else -> emptyList()
    }

    fun defaultVoice(name: String?) = when (name) {
        POCKET_NAME -> POCKET_DEFAULT_VOICE
        PHONON_NAME -> PHONON_DEFAULT_VOICE
        else -> KOKORO_DEFAULT_VOICE
    }

    /** The stored voice if this model packages it, otherwise the model's default. */
    fun resolveVoice(name: String?, stored: String?): String =
        stored?.takeIf { id -> voices(name).any { it.first == id } } ?: defaultVoice(name)

    /**
     * Locale to send with a voice. PocketTTS loads language-specific weights from it, so it must be
     * the voice's own; Kokoro derives the language from the voice id and takes none. Phonon is English only.
     */
    fun synthesisLocale(name: String?, voice: String): String? = when (name) {
        POCKET_NAME -> pocketVoiceLocales[voice]
        PHONON_NAME -> "en-US"
        else -> null
    }
}

/** Speech recognition models that do not run on WhisperKit. */
object AsrModels {
    const val PARAKEET_NAME = "Parakeet TDT 0.6B v3 (LiteRT-LM)"
    const val PARAKEET_URL =
        "https://huggingface.co/litert-community/parakeet-tdt-0.6b-v3/resolve/3ce205aa928f443f423214195cfe83ebe4802826/parakeet_tdt_0.6b_v3_5s_i8_stateful.litertlm"
    const val PARAKEET_SIZE = 614588752L

    /** OpenAI Whisper Tiny, int8, in Google's LiteRT-LM bundle; the bundle transcribes English. */
    const val WHISPER_LITERT_NAME = "Whisper Tiny int8 (LiteRT-LM)"
    const val WHISPER_LITERT_URL =
        "https://huggingface.co/litert-community/whisper-tiny/resolve/9d0bc311c487b282cd37dc846a0a75624e0bcd74/whisper_tiny_30s_i8.litertlm"
    const val WHISPER_LITERT_SIZE = 41804416L

    /** Moonshine Tiny (English). Its GPU path returns empty text on the Tab S9+, so it runs on the CPU. */
    const val MOONSHINE_NAME = "Moonshine Tiny (LiteRT-LM)"
    const val MOONSHINE_URL =
        "https://huggingface.co/litert-community/moonshine-tiny/resolve/152d4b5527c532b79cf264a6cf414b7a07978c0b/moonshine_tiny_5s_i8.litertlm"
    const val MOONSHINE_SIZE = 29676624L

    /**
     * FUTO's audio-context fine-tuned Whisper (English), 10-second window, as `.litertlm` bundles we built
     * from litert-community/whisper-acft (see timmyboy26/whisper-acft-en-litertlm): short recordings are not
     * padded to Whisper's usual 30 seconds.
     */
    private const val ACFT_BASE = "https://huggingface.co/timmyboy26/whisper-acft-en-litertlm/resolve/8d44e9fef5a82119d95ce99400ee119c3b66ff9a"
    const val ACFT_TINY_NAME = "Whisper ACFT Tiny English (LiteRT-LM)"
    const val ACFT_TINY_URL = "$ACFT_BASE/whisper_acft_tiny_en_10s.litertlm"
    const val ACFT_TINY_SIZE = 59905680L
    const val ACFT_BASE_NAME = "Whisper ACFT Base English (LiteRT-LM)"
    const val ACFT_BASE_URL = "$ACFT_BASE/whisper_acft_base_en_10s.litertlm"
    const val ACFT_BASE_SIZE = 102045960L
    const val ACFT_SMALL_NAME = "Whisper ACFT Small English (LiteRT-LM)"
    const val ACFT_SMALL_URL = "$ACFT_BASE/whisper_acft_small_en_10s.litertlm"
    const val ACFT_SMALL_SIZE = 286931488L

    /**
     * Recognizers that can transcribe while recording, with text arriving every few seconds or sooner:
     * Parakeet TDT streams with state between 5-second windows, Moonshine works in 5-second windows and ML Kit
     * is a streaming recognizer. The Whisper models wait for a 10 or 30-second window, and the Parakeet NPU
     * engine takes whole recordings, so they transcribe after the recording stops.
     */
    fun supportsLive(model: LLMModel?) =
        model != null && (model.name == PARAKEET_NAME || model.name == MOONSHINE_NAME || isMlKit(model))

    /** English-only recognizers that load in about a second. */
    fun isQuickEnglish(model: LLMModel) = model.name in setOf(MOONSHINE_NAME, WHISPER_LITERT_NAME, ACFT_TINY_NAME, ACFT_BASE_NAME, ACFT_SMALL_NAME)

    /** Model format of the system-managed ML Kit recognizers: no file of ours, AICore owns the weights. */
    const val MLKIT_FORMAT = "mlkit_asr"
    const val MLKIT_BASIC_NAME = "ML Kit Speech Recognition (Basic)"
    const val MLKIT_ADVANCED_NAME = "ML Kit Speech Recognition (Advanced)"
    const val MLKIT_BASIC_URL = "mlkit://asr/basic"
    const val MLKIT_ADVANCED_URL = "mlkit://asr/advanced"

    fun isMlKit(model: LLMModel?) = model?.modelFormat == MLKIT_FORMAT
    fun isMlKitAdvanced(model: LLMModel) = model.url == MLKIT_ADVANCED_URL
    /** Every `.litertlm` recognizer (Parakeet, Whisper, Moonshine): they all run on the MediaPipe AudioTranscriber task. */
    fun isParakeet(model: LLMModel?) = model?.category == "asr" && model.modelFormat == "litertlm"

    /** Parakeet compiled ahead of time for one SoC's NPU: a bare `.tflite` run by ParakeetNpuEngine. */
    const val PARAKEET_NPU_FORMAT = "tflite"

    /** Google's builds. */
    private const val PARAKEET_NPU_GOOGLE =
        "https://huggingface.co/litert-community/parakeet-tdt-0.6b-v3/resolve/3ce205aa928f443f423214195cfe83ebe4802826"

    /**
     * Our builds of the same model, for chips where Google's do not transcribe. The encoder's
     * LayerNorm inputs overflow float16 on those NPUs; android/tools/parakeet_npu scales them down
     * first (LayerNorm is scale-invariant) and recompiles.
     */
    private const val PARAKEET_NPU_FIXED =
        "https://huggingface.co/timmyboy26/parakeet-tdt-0.6b-v3-LiteRT-NPU/resolve/ce9aa1ac52c81e167f80344fc60596df7e2c4dd2"

    /** One NPU build: the file's SoC tag, the chip's name, the file size and the repo it comes from. */
    data class ParakeetNpuBuild(val tag: String, val chip: String, val sizeBytes: Long, val repo: String = PARAKEET_NPU_GOOGLE) {
        val name get() = "Parakeet TDT 0.6B v3 (NPU $chip)"
        val url get() = "$repo/parakeet_tdt_0.6b_v3_5s_f32_stateful_$tag.tflite"
    }

    // Every build here transcribed a spoken passage word for word on a real device, except Tensor G5,
    // which the owner asked to keep (it failed to load on a Pixel 10 with July 2025 firmware).
    //   Google's builds: Snapdragon 8 Gen 2 (Tab S9+), 8 Gen 3 (Galaxy S24), Tensor G6 (Pixel 11).
    //   Our builds: Snapdragon 8 Elite (Galaxy S25), 8 Elite Gen 5 (Galaxy S26), Dimensity 9400
    //   (Tab S11), Dimensity 9500 (Tab S12 Ultra). Google's builds for these four return NaN or a
    //   constant from the encoder.
    // Left out: Snapdragon 8 Gen 1 fails with either build (167 s for 13 s of audio, empty transcript).
    // Our builds for nine more MediaTek chips are in the repo but have not run on a device.
    val parakeetNpuBuilds = listOf(
        ParakeetNpuBuild("Qualcomm_SM8550", "Snapdragon 8 Gen 2", 1274838752L),
        ParakeetNpuBuild("Qualcomm_SM8650", "Snapdragon 8 Gen 3", 1274834656L),
        ParakeetNpuBuild("Qualcomm_SM8750", "Snapdragon 8 Elite", 1273652872L, PARAKEET_NPU_FIXED),
        ParakeetNpuBuild("Qualcomm_SM8850", "Snapdragon 8 Elite Gen 5", 1275852424L, PARAKEET_NPU_FIXED),
        ParakeetNpuBuild("Google_Tensor_G5", "Google Tensor G5", 1292830512L),
        ParakeetNpuBuild("Google_Tensor_G6", "Google Tensor G6", 1303768096L),
        ParakeetNpuBuild("MediaTek_MT6991", "MediaTek Dimensity 9400", 1255987308L, PARAKEET_NPU_FIXED),
        ParakeetNpuBuild("MediaTek_MT6993", "MediaTek Dimensity 9500", 1257704612L, PARAKEET_NPU_FIXED),
    )

    /** The NPU build compiled for [soc] (Build.SOC_MODEL), or null when none works on that chip. */
    fun parakeetNpuBuild(soc: String): ParakeetNpuBuild? {
        val chip = soc.uppercase().replace(Regex("[^A-Z0-9]"), "")
        return parakeetNpuBuilds.firstOrNull { build ->
            val id = build.tag.substringAfter('_').uppercase().replace("_", "")
            // Qualcomm appends a "P" to the same chip's faster bin; MediaTek adds a revision suffix.
            if (build.tag.startsWith("MediaTek")) chip.startsWith(id) else chip == id || chip == id + "P"
        }
    }

    fun isParakeetNpu(model: LLMModel?) = model?.category == "asr" && model.modelFormat == PARAKEET_NPU_FORMAT
}
