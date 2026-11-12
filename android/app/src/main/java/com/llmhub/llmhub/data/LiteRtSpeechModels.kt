package com.llmhub.llmhub.data

import android.content.Context
import java.io.File

/**
 * Google's `.litertlm` speech bundles (the formats AI Edge Gallery ships). Each is one file with the
 * voices packaged inside, run by the MediaPipe `tasks-audio` TextToSpeech / AudioTranscriber tasks.
 */
object LiteRtTtsModel {
    const val KOKORO_NAME = "Kokoro-82M (LiteRT-LM)"
    const val POCKET_NAME = "Pocket TTS (LiteRT-LM)"

    const val KOKORO_URL =
        "https://huggingface.co/litert-community/Kokoro-82M/resolve/81e43a025bccabcbbd487f08bc3ebd65b5fc7295/kokoro_int8_gpu_fp16.litertlm"
    const val KOKORO_SIZE = 294750530L
    const val POCKET_URL =
        "https://huggingface.co/litert-community/pocket-tts/resolve/bca93d6756489f9403f94d66826cfa65f80023a1/pocket_tts_int8.litertlm"
    const val POCKET_SIZE = 760061960L

    const val KOKORO_DEFAULT_VOICE = "af_heart"
    const val POCKET_DEFAULT_VOICE = "alba"

    fun isModel(name: String?) = name == KOKORO_NAME || name == POCKET_NAME
    fun isPocket(name: String?) = name == POCKET_NAME

    /** Only the Kokoro bundle has a GPU build; PocketTTS is CPU only. */
    fun supportsGpu(name: String?) = name == KOKORO_NAME

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
        else -> emptyList()
    }

    fun defaultVoice(name: String?) = if (name == POCKET_NAME) POCKET_DEFAULT_VOICE else KOKORO_DEFAULT_VOICE

    /** The stored voice if this model packages it, otherwise the model's default. */
    fun resolveVoice(name: String?, stored: String?): String =
        stored?.takeIf { id -> voices(name).any { it.first == id } } ?: defaultVoice(name)

    /**
     * Locale to send with a voice. PocketTTS loads language-specific weights from it, so it must be
     * the voice's own; Kokoro derives the language from the voice id and takes none.
     */
    fun synthesisLocale(name: String?, voice: String): String? =
        if (name == POCKET_NAME) pocketVoiceLocales[voice] else null
}

/** Speech recognition models that do not run on WhisperKit. */
object AsrModels {
    const val PARAKEET_NAME = "Parakeet TDT 0.6B v3 (LiteRT-LM)"
    const val PARAKEET_URL =
        "https://huggingface.co/litert-community/parakeet-tdt-0.6b-v3/resolve/3ce205aa928f443f423214195cfe83ebe4802826/parakeet_tdt_0.6b_v3_5s_i8_stateful.litertlm"
    const val PARAKEET_SIZE = 614588752L

    /** Model format of the system-managed ML Kit recognizers: no file of ours, AICore owns the weights. */
    const val MLKIT_FORMAT = "mlkit_asr"
    const val MLKIT_BASIC_NAME = "ML Kit Speech Recognition (Basic)"
    const val MLKIT_ADVANCED_NAME = "ML Kit Speech Recognition (Advanced)"
    const val MLKIT_BASIC_URL = "mlkit://asr/basic"
    const val MLKIT_ADVANCED_URL = "mlkit://asr/advanced"

    fun isMlKit(model: LLMModel?) = model?.modelFormat == MLKIT_FORMAT
    fun isMlKitAdvanced(model: LLMModel) = model.url == MLKIT_ADVANCED_URL
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
