package com.llmhub.llmhub.data

import android.os.Build

/**
 * Device SOC detection for optimal model selection
 */
object DeviceInfo {
    fun getDeviceSoc(): String {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            Build.SOC_MODEL
        } else {
            "UNKNOWN"
        }
    }
    
    private val chipsetModelSuffixes = mapOf(
        "SM8475" to "8gen1",  // 8+ Gen 1
        "SM8450" to "8gen1",  // 8 Gen 1
        "SM8550" to "8gen2",  // 8 Gen 2
        "SM8550P" to "8gen2",
        "QCS8550" to "8gen2",
        "QCM8550" to "8gen2",
        "SM8650" to "8gen3",  // 8 Gen 3 (updated)
        "SM8650P" to "8gen3",
        "SM8750" to "8gen4",  // 8 Elite / Gen 4 (updated)
        "SM8750P" to "8gen4",
        "SM8850" to "8gen4",  // Hypothetical future
        "SM8850P" to "8gen4",
        "SM8735" to "8gen3",  // 8s Gen 3
        "SM8845" to "8gen4"
    )
    
    fun getChipsetSuffix(): String? {
        val soc = getDeviceSoc()
        return chipsetModelSuffixes[soc] ?: if (soc.startsWith("SM")) "min" else null
    }
    
    fun isQualcommNpuSupported(): Boolean {
        // NPU supported on 8 Gen 2+ typically, definitely on Gen 3/4
        val suffix = getChipsetSuffix()
        return suffix == "8gen2" || suffix == "8gen3" || suffix == "8gen4"
    }
}

object ModelData {
    val models = listOf(
        // Gemma-3 1B Models
        LLMModel(
            name = "Gemma-3 1B (INT4, 2k)",
            description = "Google Gemma-3 1B with INT4 quantization and a 2k context window. Optimized for mobile devices. Ready to download from HuggingFace (529MB)",
            url = "https://huggingface.co/litert-community/Gemma3-1B-IT/resolve/main/Gemma3-1B-IT_multi-prefill-seq_q4_ekv2048.task?download=true",
            category = "text",
            sizeBytes = 554661246L, // 529MB (actual size from HuggingFace)
            source = "Google via LiteRT Community",
            supportsVision = false,
            supportsGpu = true,
            requirements = ModelRequirements(minRamGB = 2, recommendedRamGB = 3),
            contextWindowSize = 2048,
            modelFormat = "task"
        ),
        LLMModel(
            name = "Gemma-3 1B (INT8, 1.2k)",
            description = "Higher quality INT8 version of Gemma-3 1B with a 1.2k context window. Ready to download from HuggingFace (1005MB)",
            url = "https://huggingface.co/litert-community/Gemma3-1B-IT/resolve/main/Gemma3-1B-IT_multi-prefill-seq_q8_ekv1280.task?download=true",
            category = "text",
            sizeBytes = 1054012582L, // 1005MB (actual size from HuggingFace)
            source = "Google via LiteRT Community",
            supportsVision = false,
            supportsGpu = true,
            requirements = ModelRequirements(minRamGB = 3, recommendedRamGB = 4),
            contextWindowSize = 1280,
            modelFormat = "task"
        ),
        LLMModel(
            name = "Gemma-3 1B (INT8, 2k)",
            description = "Higher quality INT8 version of Gemma-3 1B with a 2k context window. Ready to download from HuggingFace (1024MB)",
            url = "https://huggingface.co/litert-community/Gemma3-1B-IT/resolve/main/Gemma3-1B-IT_multi-prefill-seq_q8_ekv2048.task?download=true",
            category = "text",
            sizeBytes = 1073765694L, // 1024MB (actual size from HuggingFace)
            source = "Google via LiteRT Community",
            supportsVision = false,
            supportsGpu = true,
            requirements = ModelRequirements(minRamGB = 3, recommendedRamGB = 4),
            contextWindowSize = 2048,
            modelFormat = "task"
        ),
        LLMModel(
            name = "Gemma-3 1B (INT8, 4k)",
            description = "Higher quality INT8 version of Gemma-3 1B with a large 4k context window. Ready to download from HuggingFace (1005MB)",
            url = "https://huggingface.co/litert-community/Gemma3-1B-IT/resolve/main/Gemma3-1B-IT_multi-prefill-seq_q8_ekv4096.task?download=true",
            category = "text",
            sizeBytes = 1054023846L, // 1005MB (actual size from HuggingFace)
            source = "Google via LiteRT Community",
            supportsVision = false,
            supportsGpu = true,
            requirements = ModelRequirements(minRamGB = 4, recommendedRamGB = 5),
            contextWindowSize = 4096,
            modelFormat = "task"
        ),

        // Llama-3.2 Models from vimal-yuvabe
        LLMModel(
            name = "Llama-3.2 1B (INT8)",
            description = "Meta's Llama 3.2 1B model with INT8 quantization. Optimized for on-device inference. Ready to download from HuggingFace (2.01GB)",
            url = "https://huggingface.co/vimal-yuvabe/llama-3.2-1b-tflite/resolve/main/llama-3.2-1b-q8.task?download=true",
            category = "text",
            sizeBytes = 2160086757L, // 2.01GB (actual size from HuggingFace)
            source = "Meta via vimal-yuvabe",
            supportsVision = false,
            supportsGpu = false, // Llama models have GPU compatibility issues
            requirements = ModelRequirements(minRamGB = 3, recommendedRamGB = 4),
            contextWindowSize = 4096,
            modelFormat = "task"
        ),
        
        LLMModel(
            name = "Llama-3.2 3B (INT8)",
            description = "Meta's Llama 3.2 3B model with INT8 quantization. Larger model for better performance. Ready to download from HuggingFace (5.11GB)",
            url = "https://huggingface.co/vimal-yuvabe/llama-3.2-3b-tflite/resolve/main/llama-3.2-3B-q8.task?download=true",
            category = "text", 
            sizeBytes = 5491473637L, // 5.11GB (actual size from HuggingFace)
            source = "Meta via vimal-yuvabe",
            supportsVision = false,
            supportsGpu = false,
            requirements = ModelRequirements(minRamGB = 6, recommendedRamGB = 8),
            contextWindowSize = 4096,
            modelFormat = "task"
        ),

        // Phi-4 Mini updated to user-provided litertlm link
        LLMModel(
            name = "Phi-4 Mini (INT8, 4k)",
            description = "Phi-4 Mini INT8 variant in LiteRT LM format (litertlm).",
            url = "https://huggingface.co/litert-community/Phi-4-mini-instruct/resolve/main/Phi-4-mini-instruct_multi-prefill-seq_q8_ekv4096.litertlm?download=true",
            category = "text",
            sizeBytes = 3910090752L, // 3.91 GB as reported on HuggingFace file page
            source = "Microsoft via LiteRT Community",
            supportsVision = false,
            supportsGpu = true,
            requirements = ModelRequirements(minRamGB = 5, recommendedRamGB = 7),
            contextWindowSize = 4096,
            modelFormat = "litertlm"
        ),

        // LFM-2.5 1.2B Instruct Models (LiquidAI GGUF)
        LLMModel(
            name = "LFM-2.5 1.2B Instruct (Q4_0)",
            description = "LiquidAI's 1.2B instruct model. Q4_0 quantization. 128k context.",
            url = "https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF/resolve/main/LFM2.5-1.2B-Instruct-Q4_0.gguf?download=true",
            category = "text",
            sizeBytes = 696000000L, // 696 MB (actual HF file size)
            source = "LiquidAI",
            supportsVision = false,
            supportsGpu = true,
            requirements = ModelRequirements(minRamGB = 2, recommendedRamGB = 3),
            contextWindowSize = 128000,
            modelFormat = "gguf"
        ),
        LLMModel(
            name = "LFM-2.5 1.2B Instruct (Q4_K_M)",
            description = "LiquidAI's 1.2B instruct model. Q4_K_M quantization. 128k context.",
            url = "https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF/resolve/main/LFM2.5-1.2B-Instruct-Q4_K_M.gguf?download=true",
            category = "text",
            sizeBytes = 731000000L, // 731 MB (actual HF file size)
            source = "LiquidAI",
            supportsVision = false,
            supportsGpu = true,
            requirements = ModelRequirements(minRamGB = 2, recommendedRamGB = 3),
            contextWindowSize = 128000,
            modelFormat = "gguf"
        ),
        LLMModel(
            name = "LFM-2.5 1.2B Instruct (Q8_0)",
            description = "LiquidAI's 1.2B instruct model. Q8_0 quantization. 128k context.",
            url = "https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF/resolve/main/LFM2.5-1.2B-Instruct-Q8_0.gguf?download=true",
            category = "text",
            sizeBytes = 1250000000L, // 1.25 GB (actual HF file size)
            source = "LiquidAI",
            supportsVision = false,
            supportsGpu = true,
            requirements = ModelRequirements(minRamGB = 3, recommendedRamGB = 4),
            contextWindowSize = 128000,
            modelFormat = "gguf"
        ),
        
        // LFM-2.5 1.2B Thinking Models (LiquidAI GGUF)
        LLMModel(
            name = "LFM-2.5 1.2B Thinking (Q4_0)",
            description = "LiquidAI's 1.2B thinking model. Q4_0 quantization. 128k context. Supports 'thinking' mode.",
            url = "https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking-GGUF/resolve/main/LFM2.5-1.2B-Thinking-Q4_0.gguf?download=true",
            category = "text",
            sizeBytes = 696000000L, // 696 MB (actual HF file size)
            source = "LiquidAI",
            supportsVision = false,
            supportsGpu = true,
            requirements = ModelRequirements(minRamGB = 2, recommendedRamGB = 3),
            contextWindowSize = 128000,
            modelFormat = "gguf"
        ),
        LLMModel(
            name = "LFM-2.5 1.2B Thinking (Q4_K_M)",
            description = "LiquidAI's 1.2B thinking model. Q4_K_M quantization. 128k context. Supports 'thinking' mode.",
            url = "https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking-GGUF/resolve/main/LFM2.5-1.2B-Thinking-Q4_K_M.gguf?download=true",
            category = "text",
            sizeBytes = 731000000L, // 731 MB (actual HF file size)
            source = "LiquidAI",
            supportsVision = false,
            supportsGpu = true,
            requirements = ModelRequirements(minRamGB = 2, recommendedRamGB = 3),
            contextWindowSize = 128000,
            modelFormat = "gguf"
        ),
        LLMModel(
            name = "LFM-2.5 1.2B Thinking (Q8_0)",
            description = "LiquidAI's 1.2B thinking model. Q8_0 quantization. 128k context. Supports 'thinking' mode.",
            url = "https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking-GGUF/resolve/main/LFM2.5-1.2B-Thinking-Q8_0.gguf?download=true",
            category = "text",
            sizeBytes = 1250000000L, // 1.25 GB (actual HF file size)
            source = "LiquidAI",
            supportsVision = false,
            supportsGpu = true,
            requirements = ModelRequirements(minRamGB = 3, recommendedRamGB = 4),
            contextWindowSize = 128000,
            modelFormat = "gguf"
        ),

        // LFM-2.5 1.2B Instruct Models (ONNX) - Q4 and Q8 variants (sizeBytes = sum of all downloaded files from HF)
        LLMModel(
            name = "LFM-2.5 1.2B Instruct (ONNX Q4)",
            description = "LiquidAI's 1.2B instruction model in ONNX format. Q4 quantization for balanced quality and size. 128k context. Requires downloading 2 files.",
            url = "https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct-ONNX/resolve/main/onnx/model_q4.onnx?download=true",
            category = "text",
            sizeBytes = 1224116481L, // main + onnx_data + tokenizer.json + tokenizer_config.json (HF API)
            source = "LiquidAI",
            supportsVision = false,
            supportsGpu = false,
            requirements = ModelRequirements(minRamGB = 2, recommendedRamGB = 4),
            contextWindowSize = 128000,
            modelFormat = "onnx",
            additionalFiles = listOf(
                "https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct-ONNX/resolve/main/onnx/model_q4.onnx_data?download=true",
                "https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct-ONNX/resolve/main/tokenizer.json?download=true",
                "https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct-ONNX/resolve/main/tokenizer_config.json?download=true"
            )
        ),
        LLMModel(
            name = "LFM-2.5 1.2B Instruct (ONNX Q8)",
            description = "LiquidAI's 1.2B instruction model in ONNX format. Q8 quantization for higher quality. 128k context. Requires downloading 2 files.",
            url = "https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct-ONNX/resolve/main/onnx/model_q8.onnx?download=true",
            category = "text",
            sizeBytes = 1772844567L, // main + onnx_data + tokenizer (HF API)
            source = "LiquidAI",
            supportsVision = false,
            supportsGpu = false,
            requirements = ModelRequirements(minRamGB = 3, recommendedRamGB = 5),
            contextWindowSize = 128000,
            modelFormat = "onnx",
            additionalFiles = listOf(
                "https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct-ONNX/resolve/main/onnx/model_q8.onnx_data?download=true",
                "https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct-ONNX/resolve/main/tokenizer.json?download=true",
                "https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct-ONNX/resolve/main/tokenizer_config.json?download=true"
            )
        ),

        // LFM-2.5 1.2B Thinking Models (ONNX) - Q4 and Q8 variants (sizeBytes = sum of all downloaded files from HF)
        LLMModel(
            name = "LFM-2.5 1.2B Thinking (ONNX Q4)",
            description = "LiquidAI's 1.2B reasoning model in ONNX format. Q4 quantization for balanced quality and size. 128k context. Requires downloading 2 files.",
            url = "https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking-ONNX/resolve/main/onnx/model_q4.onnx?download=true",
            category = "text",
            sizeBytes = 1224116481L, // main + onnx_data + tokenizer (HF API; Thinking repo same layout)
            source = "LiquidAI",
            supportsVision = false,
            supportsGpu = false,
            requirements = ModelRequirements(minRamGB = 2, recommendedRamGB = 4),
            contextWindowSize = 128000,
            modelFormat = "onnx",
            additionalFiles = listOf(
                "https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking-ONNX/resolve/main/onnx/model_q4.onnx_data?download=true",
                "https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking-ONNX/resolve/main/tokenizer.json?download=true",
                "https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking-ONNX/resolve/main/tokenizer_config.json?download=true"
            )
        ),
        LLMModel(
            name = "LFM-2.5 1.2B Thinking (ONNX Q8)",
            description = "LiquidAI's 1.2B reasoning model in ONNX format. Q8 quantization for higher quality. 128k context. Requires downloading 2 files.",
            url = "https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking-ONNX/resolve/main/onnx/model_q8.onnx?download=true",
            category = "text",
            sizeBytes = 1772844567L, // main + onnx_data + tokenizer (HF API)
            source = "LiquidAI",
            supportsVision = false,
            supportsGpu = false,
            requirements = ModelRequirements(minRamGB = 3, recommendedRamGB = 5),
            contextWindowSize = 128000,
            modelFormat = "onnx",
            additionalFiles = listOf(
                "https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking-ONNX/resolve/main/onnx/model_q8.onnx_data?download=true",
                "https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking-ONNX/resolve/main/tokenizer.json?download=true",
                "https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking-ONNX/resolve/main/tokenizer_config.json?download=true"
            )
        ),

        // LFM-2.5 VL 1.6B Models (Vision-Language GGUF)
        LLMModel(
            name = "LFM-2.5 VL 1.6B (BF16)",
            description = "LiquidAI's 1.6B vision-language model. BF16 precision. Supports vision + text. Requires mmproj for vision.",
            url = "https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B-GGUF/resolve/main/LFM2.5-VL-1.6B-BF16.gguf?download=true",
            category = "multimodal",
            sizeBytes = 2340000000L, // 2.34 GB from HuggingFace
            source = "LiquidAI",
            supportsVision = true,
            supportsGpu = true,
            requirements = ModelRequirements(minRamGB = 4, recommendedRamGB = 6),
            contextWindowSize = 128000,
            modelFormat = "gguf"
        ),
        LLMModel(
            name = "LFM-2.5 VL 1.6B (F16)",
            description = "LiquidAI's 1.6B vision-language model. F16 precision. Supports vision + text. Requires mmproj for vision.",
            url = "https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B-GGUF/resolve/main/LFM2.5-VL-1.6B-F16.gguf?download=true",
            category = "multimodal",
            sizeBytes = 2340000000L, // 2.34 GB from HuggingFace
            source = "LiquidAI",
            supportsVision = false,
            supportsGpu = true,
            requirements = ModelRequirements(minRamGB = 4, recommendedRamGB = 6),
            contextWindowSize = 128000,
            modelFormat = "gguf"
        ),
        LLMModel(
            name = "LFM-2.5 VL 1.6B (Q4_0)",
            description = "LiquidAI's 1.6B vision-language model. Q4_0 quantization. Supports vision + text. Requires mmproj for vision.",
            url = "https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B-GGUF/resolve/main/LFM2.5-VL-1.6B-Q4_0.gguf?download=true",
            category = "multimodal",
            sizeBytes = 696000000L, // 696 MB from HuggingFace
            source = "LiquidAI",
            supportsVision = false,
            supportsGpu = true,
            requirements = ModelRequirements(minRamGB = 3, recommendedRamGB = 4),
            contextWindowSize = 128000,
            modelFormat = "gguf"
        ),
        LLMModel(
            name = "LFM-2.5 VL 1.6B (Q8_0)",
            description = "LiquidAI's 1.6B vision-language model. Q8_0 quantization. Supports vision + text. Requires mmproj for vision.",
            url = "https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B-GGUF/resolve/main/LFM2.5-VL-1.6B-Q8_0.gguf?download=true",
            category = "multimodal",
            sizeBytes = 1250000000L, // 1.25 GB from HuggingFace
            source = "LiquidAI",
            supportsVision = true,
            supportsGpu = true,
            requirements = ModelRequirements(minRamGB = 3, recommendedRamGB = 5),
            contextWindowSize = 128000,
            modelFormat = "gguf"
        ),
        LLMModel(
            name = "LFM-2.5 VL 1.6B (Vision Projector, BF16)",
            description = "Vision Projector for LFM-2.5 VL models. BF16 variant required for image input. Download this to enable vision.",
            url = "https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B-GGUF/resolve/main/mmproj-LFM2.5-VL-1.6b-BF16.gguf?download=true",
            category = "multimodal",
            sizeBytes = 856000000L, // 856 MB from HuggingFace
            source = "LiquidAI",
            supportsVision = true,
            supportsGpu = true,
            requirements = ModelRequirements(minRamGB = 1, recommendedRamGB = 2),
            contextWindowSize = 0,
            modelFormat = "gguf"
        ),
        LLMModel(
            name = "LFM-2.5 VL 1.6B (Vision Projector, Q8_0)",
            description = "Vision Projector for LFM-2.5 VL models. Q8_0 quantized variant for smaller size. Download this to enable vision.",
            url = "https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B-GGUF/resolve/main/mmproj-LFM2.5-VL-1.6b-Q8_0.gguf?download=true",
            category = "multimodal",
            sizeBytes = 583000000L, // 583 MB from HuggingFace
            source = "LiquidAI",
            supportsVision = true,
            supportsGpu = true,
            requirements = ModelRequirements(minRamGB = 1, recommendedRamGB = 2),
            contextWindowSize = 0,
            modelFormat = "gguf"
        ),

        // LFM-2.5 VL 1.6B Models (Vision-Language ONNX)
        // LLMModel(
        //     name = "LFM-2.5 VL 1.6B (ONNX Q4)",
        //     description = "LiquidAI's 1.6B vision-language model in ONNX format. Q4 quantization. Supports vision + text. Requires multiple files.",
        //     url = "https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B-ONNX/resolve/main/onnx/decoder_q4.onnx?download=true",
        //     category = "multimodal",
        //     sizeBytes = 1760000000L, // decoder_q4 + embed_images_q4 + embed_tokens_fp16 + tokenizer (~1.76 GB)
        //     source = "LiquidAI",
        //     supportsVision = true,
        //     supportsGpu = false,
        //     requirements = ModelRequirements(minRamGB = 3, recommendedRamGB = 5),
        //     contextWindowSize = 128000,
        //     modelFormat = "onnx",
        //     additionalFiles = listOf(
        //         "https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B-ONNX/resolve/main/onnx/decoder_q4.onnx_data?download=true",
        //         "https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B-ONNX/resolve/main/onnx/embed_images_q4.onnx?download=true",
        //         "https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B-ONNX/resolve/main/onnx/embed_images_q4.onnx_data?download=true",
        //         "https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B-ONNX/resolve/main/onnx/embed_tokens_fp16.onnx?download=true",
        //         "https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B-ONNX/resolve/main/onnx/embed_tokens_fp16.onnx_data?download=true",
        //         "https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B-ONNX/resolve/main/config.json?download=true",
        //         "https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B-ONNX/resolve/main/tokenizer.json?download=true",
        //         "https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B-ONNX/resolve/main/tokenizer_config.json?download=true"
        //     )
        // ),
        // LLMModel(
        //     name = "LFM-2.5 VL 1.6B (ONNX Q8)",
        //     description = "LiquidAI's 1.6B vision-language model in ONNX format. Q8 quantization for higher quality. Supports vision + text.",
        //     url = "https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B-ONNX/resolve/main/onnx/decoder_q8.onnx?download=true",
        //     category = "multimodal",
        //     sizeBytes = 2540000000L, // decoder_q8 + embed_images_q8 + embed_tokens_fp16 + tokenizer (~2.54 GB)
        //     source = "LiquidAI",
        //     supportsVision = true,
        //     supportsGpu = false,
        //     requirements = ModelRequirements(minRamGB = 4, recommendedRamGB = 6),
        //     contextWindowSize = 128000,
        //     modelFormat = "onnx",
        //     additionalFiles = listOf(
        //         "https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B-ONNX/resolve/main/onnx/decoder_q8.onnx_data?download=true",
        //         "https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B-ONNX/resolve/main/onnx/embed_images_q8.onnx?download=true",
        //         "https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B-ONNX/resolve/main/onnx/embed_images_q8.onnx_data?download=true",
        //         "https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B-ONNX/resolve/main/onnx/embed_tokens_fp16.onnx?download=true",
        //         "https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B-ONNX/resolve/main/onnx/embed_tokens_fp16.onnx_data?download=true",
        //         "https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B-ONNX/resolve/main/config.json?download=true",
        //         "https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B-ONNX/resolve/main/tokenizer.json?download=true",
        //         "https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B-ONNX/resolve/main/tokenizer_config.json?download=true"
        //     )
        // ),

        // Ministral-3 3B Instruct Models (MistralAI GGUF)
        LLMModel(
            name = "Ministral-3 3B Instruct (Q4_K_M)",
            description = "MistralAI's 3B instruct model. Q4_K_M quantization. 32k context. Supports Vision (Requires mmproj).",
            url = "https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512-GGUF/resolve/main/Ministral-3-3B-Instruct-2512-Q4_K_M.gguf?download=true",
            category = "multimodal",
            sizeBytes = 2150000000L, // 2.15 GB from HuggingFace
            source = "MistralAI",
            supportsVision = true,
            supportsGpu = true,
            requirements = ModelRequirements(minRamGB = 4, recommendedRamGB = 6),
            contextWindowSize = 262144,
            modelFormat = "gguf"
        ),
        LLMModel(
            name = "Ministral-3 3B Instruct (Q5_K_M)",
            description = "MistralAI's 3B instruct model. Q5_K_M quantization. 32k context. Supports Vision (Requires mmproj).",
            url = "https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512-GGUF/resolve/main/Ministral-3-3B-Instruct-2512-Q5_K_M.gguf?download=true",
            category = "multimodal",
            sizeBytes = 2470000000L, // 2.47 GB from HuggingFace
            source = "MistralAI",
            supportsVision = true,
            supportsGpu = true,
            requirements = ModelRequirements(minRamGB = 4, recommendedRamGB = 7),
            contextWindowSize = 262144,
            modelFormat = "gguf"
        ),
        LLMModel(
            name = "Ministral-3 3B Instruct (Q8_0)",
            description = "MistralAI's 3B instruct model. Q8_0 quantization. 32k context. Supports Vision (Requires mmproj).",
            url = "https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512-GGUF/resolve/main/Ministral-3-3B-Instruct-2512-Q8_0.gguf?download=true",
            category = "multimodal",
            sizeBytes = 3650000000L, // 3.65 GB from HuggingFace
            source = "MistralAI",
            supportsVision = true,
            supportsGpu = true,
            requirements = ModelRequirements(minRamGB = 6, recommendedRamGB = 8),
            contextWindowSize = 262144,
            modelFormat = "gguf"
        ),
        LLMModel(
            name = "Ministral-3 3B Instruct (Vision Projector, BF16)",
            description = "Multimodal Vision Projector for Ministral-3 3B models. Specifically the BF16 variant required for image input capabilities. Download this if you want to enable Vision for Ministral models.",
            url = "https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512-GGUF/resolve/main/Ministral-3-3B-Instruct-2512-BF16-mmproj.gguf?download=true",
            category = "multimodal",
            sizeBytes = 842000000L, // 842 MB from HuggingFace
            source = "MistralAI",
            supportsVision = true,
            supportsGpu = true,
            requirements = ModelRequirements(minRamGB = 1, recommendedRamGB = 2),
            contextWindowSize = 0,
            modelFormat = "gguf"
        ),

        // Ministral-3 3B Instruct Models (ONNX) - Q4/Q4F16 (sizeBytes = sum of all downloaded files from HF)
        LLMModel(
            name = "Ministral-3 3B Instruct (ONNX Q4)",
            description = "MistralAI's 3B instruction model in ONNX format. Q4 quantization - recommended for mobile. 32k context. Supports vision. Requires downloading multiple files.",
            url = "https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512-ONNX/resolve/main/onnx/decoder_model_merged_q4.onnx?download=true",
            category = "multimodal",
            sizeBytes = 3367933870L, // decoder_q4 + decoder_data x2 + embed_fp16 + vision_q4 + config + tokenizer (HF API)
            source = "MistralAI",
            supportsVision = true,
            supportsGpu = false,
            requirements = ModelRequirements(minRamGB = 4, recommendedRamGB = 6),
            contextWindowSize = 32768,
            modelFormat = "onnx",
            additionalFiles = listOf(
                "https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512-ONNX/resolve/main/onnx/decoder_model_merged_q4.onnx_data?download=true",
                "https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512-ONNX/resolve/main/onnx/decoder_model_merged_q4.onnx_data_1?download=true",
                "https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512-ONNX/resolve/main/onnx/embed_tokens_fp16.onnx?download=true",
                "https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512-ONNX/resolve/main/onnx/embed_tokens_fp16.onnx_data?download=true",
                "https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512-ONNX/resolve/main/onnx/vision_encoder_q4.onnx?download=true",
                "https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512-ONNX/resolve/main/onnx/vision_encoder_q4.onnx_data?download=true",
                "https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512-ONNX/resolve/main/config.json?download=true",
                "https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512-ONNX/resolve/main/tokenizer.json?download=true",
                "https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512-ONNX/resolve/main/tokenizer_config.json?download=true"
            )
        ),
        LLMModel(
            name = "Ministral-3 3B Instruct (ONNX Q4F16)",
            description = "MistralAI's 3B instruction model in ONNX format. Q4F16 mixed precision - best quality/size balance. 32k context. Supports vision.",
            url = "https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512-ONNX/resolve/main/onnx/decoder_model_merged_q4f16.onnx?download=true",
            category = "multimodal",
            sizeBytes = 3007199181L, // decoder_q4f16 + decoder_data + embed_fp16 + vision_q4 + config + tokenizer (HF API)
            source = "MistralAI",
            supportsVision = true,
            supportsGpu = false,
            requirements = ModelRequirements(minRamGB = 4, recommendedRamGB = 6),
            contextWindowSize = 32768,
            modelFormat = "onnx",
            additionalFiles = listOf(
                "https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512-ONNX/resolve/main/onnx/decoder_model_merged_q4f16.onnx_data?download=true",
                "https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512-ONNX/resolve/main/onnx/embed_tokens_fp16.onnx?download=true",
                "https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512-ONNX/resolve/main/onnx/embed_tokens_fp16.onnx_data?download=true",
                "https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512-ONNX/resolve/main/onnx/vision_encoder_q4.onnx?download=true",
                "https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512-ONNX/resolve/main/onnx/vision_encoder_q4.onnx_data?download=true",
                "https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512-ONNX/resolve/main/config.json?download=true",
                "https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512-ONNX/resolve/main/tokenizer.json?download=true",
                "https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512-ONNX/resolve/main/tokenizer_config.json?download=true"
            )
        ),

        // Gemma-3n Models (Multimodal - Text + Vision + Audio)
        LLMModel(
            name = "Gemma-3n E2B",
            description = "Google Gemma-3n E2B with multimodal capabilities (text, vision, and audio). Effective 2B parameters with selective parameter activation. Supports 4k context window and multimodal input including text, images, and audio. Ready to download from HuggingFace (3.1.15GB)",
            url = "https://huggingface.co/google/gemma-3n-E2B-it-litert-lm/resolve/73b019b63436d346f68dd9c1dbfd117eb264d888/gemma-3n-E2B-it-int4.litertlm?download=true",
            category = "multimodal",
            sizeBytes = 3388604416L, // 3.1.15GB (actual downloaded size)
            source = "Google (LiteRT LM)",
            supportsVision = true,
            supportsAudio = true,
            supportsGpu = true,
            requirements = ModelRequirements(minRamGB = 4, recommendedRamGB = 6),
            contextWindowSize = 4096,
            modelFormat = "litertlm"
        ),
        LLMModel(
            name = "Gemma-3n E4B",
            description = "Google Gemma-3n E4B with multimodal capabilities (text, vision, and audio). Effective 4B parameters with selective parameter activation. Supports 4k context window and multimodal input including text, images, and audio. Ready to download from HuggingFace (4.33GB)",
            url = "https://huggingface.co/google/gemma-3n-E4B-it-litert-lm/resolve/3d0179a0648381585ab337e170b7517aae8e0ce4/gemma-3n-E4B-it-int4.litertlm?download=true",
            category = "multimodal",
            sizeBytes = 4652318720L, // 4.33GB (actual downloaded size)
            source = "Google (LiteRT LM)",
            supportsVision = true,
            supportsAudio = true,
            supportsGpu = true,
            requirements = ModelRequirements(minRamGB = 6, recommendedRamGB = 7),
            contextWindowSize = 4096,
            modelFormat = "litertlm"
        ),
        
        // Gecko Embedding Models - Various dimension sizes for different use cases
        LLMModel(
            name = "Gecko-110M (64D Quantized)",
            description = "Compact Gecko embedding model with 64 dimensions, quantized for minimal storage and fast inference.",
            url = "https://huggingface.co/litert-community/Gecko-110m-en/resolve/main/Gecko_64_quant.tflite?download=true",
            category = "embedding",
            sizeBytes = 112175104L, // 106.98 MB
            source = "Google via LiteRT Community",
            supportsVision = false,
            supportsGpu = false,
            requirements = ModelRequirements(minRamGB = 1, recommendedRamGB = 1),
            contextWindowSize = 64,
            modelFormat = "tflite"
        ),
        LLMModel(
            name = "Gecko-110M (64D Float32)",
            description = "Gecko embedding model with 64 dimensions in full precision for highest quality with small vectors.",
            url = "https://huggingface.co/litert-community/Gecko-110m-en/resolve/main/Gecko_64_f32.tflite?download=true",
            category = "embedding",
            sizeBytes = 441231836L, // 420.79 MB
            source = "Google via LiteRT Community",
            supportsVision = false,
            supportsGpu = false,
            requirements = ModelRequirements(minRamGB = 2, recommendedRamGB = 3),
            contextWindowSize = 64,
            modelFormat = "tflite"
        ),
        LLMModel(
            name = "Gecko-110M (256D Quantized)",
            description = "Balanced Gecko embedding model with 256 dimensions, quantized for good quality and reasonable size.",
            url = "https://huggingface.co/litert-community/Gecko-110m-en/resolve/main/Gecko_256_quant.tflite?download=true",
            category = "embedding",
            sizeBytes = 114141184L, // 108.85 MB
            source = "Google via LiteRT Community",
            supportsVision = false,
            supportsGpu = false,
            requirements = ModelRequirements(minRamGB = 1, recommendedRamGB = 2),
            contextWindowSize = 256,
            modelFormat = "tflite"
        ),
        LLMModel(
            name = "Gecko-110M (256D Float32)",
            description = "High-quality Gecko embedding model with 256 dimensions in full precision.",
            url = "https://huggingface.co/litert-community/Gecko-110m-en/resolve/main/Gecko_256_f32.tflite?download=true",
            category = "embedding",
            sizeBytes = 443197916L, // 422.67 MB
            source = "Google via LiteRT Community",
            supportsVision = false,
            supportsGpu = false,
            requirements = ModelRequirements(minRamGB = 2, recommendedRamGB = 3),
            contextWindowSize = 256,
            modelFormat = "tflite"
        ),
        LLMModel(
            name = "Gecko-110M (512D Quantized)",
            description = "High-dimensional Gecko embedding model with 512 dimensions, quantized for balanced performance.",
            url = "https://huggingface.co/litert-community/Gecko-110m-en/resolve/main/Gecko_512_quant.tflite?download=true",
            category = "embedding",
            sizeBytes = 120432640L, // 114.85 MB
            source = "Google via LiteRT Community",
            supportsVision = false,
            supportsGpu = false,
            requirements = ModelRequirements(minRamGB = 1, recommendedRamGB = 2),
            contextWindowSize = 512,
            modelFormat = "tflite"
        ),
        LLMModel(
            name = "Gecko-110M (512D Float32)",
            description = "Premium Gecko embedding model with 512 dimensions in full precision for best semantic understanding.",
            url = "https://huggingface.co/litert-community/Gecko-110m-en/resolve/main/Gecko_512_f32.tflite?download=true",
            category = "embedding",
            sizeBytes = 449489372L, // 428.67 MB
            source = "Google via LiteRT Community",
            supportsVision = false,
            supportsGpu = false,
            requirements = ModelRequirements(minRamGB = 2, recommendedRamGB = 3),
            contextWindowSize = 512,
            modelFormat = "tflite"
        ),
        LLMModel(
            name = "Gecko-110M (1024D Quantized)",
            description = "Maximum dimension Gecko embedding model with 1024 dimensions, quantized for comprehensive semantic representation.",
            url = "https://huggingface.co/litert-community/Gecko-110m-en/resolve/main/Gecko_1024_quant.tflite?download=true",
            category = "embedding",
            sizeBytes = 145598464L, // 138.85 MB
            source = "Google via LiteRT Community",
            supportsVision = false,
            supportsGpu = false,
            requirements = ModelRequirements(minRamGB = 2, recommendedRamGB = 3),
            contextWindowSize = 1024,
            modelFormat = "tflite"
        ),
        LLMModel(
            name = "Gecko-110M (1024D Float32)",
            description = "Top-tier Gecko embedding model with 1024 dimensions in full precision for maximum semantic accuracy.",
            url = "https://huggingface.co/litert-community/Gecko-110m-en/resolve/main/Gecko_1024_f32.tflite?download=true",
            category = "embedding",
            sizeBytes = 474655196L, // 452.67 MB
            source = "Google via LiteRT Community",
            supportsVision = false,
            supportsGpu = false,
            requirements = ModelRequirements(minRamGB = 3, recommendedRamGB = 4),
            contextWindowSize = 1024,
            modelFormat = "tflite"
        ),
        
        // EmbeddingGemma Models - High-quality text embeddings from Google
        LLMModel(
            name = "EmbeddingGemma 300M (256 seq)",
            description = "Google EmbeddingGemma 300M model with 256 sequence length. High-quality text embeddings for semantic search and similarity tasks. Mixed-precision for optimal performance. Ready to download from HuggingFace (170.84MB)",
            url = "https://huggingface.co/litert-community/embeddinggemma-300m/resolve/main/embeddinggemma-300M_seq256_mixed-precision.tflite?download=true",
            category = "embedding",
            sizeBytes = 179131736L, // 170.84 MB
            source = "Google via LiteRT Community",
            supportsVision = false,
            supportsGpu = false,
            requirements = ModelRequirements(minRamGB = 2, recommendedRamGB = 3),
            contextWindowSize = 256,
            modelFormat = "tflite"
        ),
        LLMModel(
            name = "EmbeddingGemma 300M (512 seq)",
            description = "Google EmbeddingGemma 300M model with 512 sequence length. High-quality text embeddings for semantic search and similarity tasks. Mixed-precision for optimal performance. Ready to download from HuggingFace (170.84MB)",
            url = "https://huggingface.co/litert-community/embeddinggemma-300m/resolve/main/embeddinggemma-300M_seq512_mixed-precision.tflite?download=true",
            category = "embedding",
            sizeBytes = 179132472L, // 170.84 MB
            source = "Google via LiteRT Community",
            supportsVision = false,
            supportsGpu = false,
            requirements = ModelRequirements(minRamGB = 2, recommendedRamGB = 3),
            contextWindowSize = 512,
            modelFormat = "tflite"
        ),
        LLMModel(
            name = "EmbeddingGemma 300M (1024 seq)",
            description = "Google EmbeddingGemma 300M model with 1024 sequence length. High-quality text embeddings for semantic search and similarity tasks. Mixed-precision for optimal performance. Ready to download from HuggingFace (174.84MB)",
            url = "https://huggingface.co/litert-community/embeddinggemma-300m/resolve/main/embeddinggemma-300M_seq1024_mixed-precision.tflite?download=true",
            category = "embedding",
            sizeBytes = 183329528L, // 174.84 MB
            source = "Google via LiteRT Community",
            supportsVision = false,
            supportsGpu = false,
            requirements = ModelRequirements(minRamGB = 2, recommendedRamGB = 3),
            contextWindowSize = 1024,
            modelFormat = "tflite"
        ),
        LLMModel(
            name = "EmbeddingGemma 300M (2048 seq)",
            description = "Google EmbeddingGemma 300M model with 2048 sequence length. High-quality text embeddings for semantic search and similarity tasks. Mixed-precision for optimal performance. Ready to download from HuggingFace (186.84MB)",
            url = "https://huggingface.co/litert-community/embeddinggemma-300m/resolve/main/embeddinggemma-300M_seq2048_mixed-precision.tflite?download=true",
            category = "embedding",
            sizeBytes = 195912440L, // 186.84 MB
            source = "Google via LiteRT Community",
            supportsVision = false,
            supportsGpu = false,
            requirements = ModelRequirements(minRamGB = 2, recommendedRamGB = 3),
            contextWindowSize = 2048,
            modelFormat = "tflite"
        ),
        
        // Image Generation Models - Absolute Reality (Stable Diffusion 1.5)
        LLMModel(
            name = "Absolute Reality (NPU - ${DeviceInfo.getChipsetSuffix() ?: "Not Supported"})",
            description = "Absolute Reality SD1.5 model optimized for Qualcomm NPU acceleration using QNN SDK. Supports txt2img generation at 512x512 resolution. Requires Snapdragon 8 Gen 1 or newer with Hexagon NPU. Device detected: ${DeviceInfo.getDeviceSoc()}. ~1.06 GB download from HuggingFace.",
            url = "https://huggingface.co/xororz/sd-qnn/resolve/main/AbsoluteReality_qnn2.28_${DeviceInfo.getChipsetSuffix() ?: "min"}.zip",
            category = "image_generation",
            sizeBytes = when(DeviceInfo.getChipsetSuffix()) {
                "8gen1" -> 1138900992L    // 1.06 GB
                "8gen2" -> 1128267776L    // 1.05 GB  
                else -> 1041235968L       // 993 MB (min)
            },
            source = "Stable Diffusion 1.5 (QNN SDK via xororz)",
            supportsVision = false,
            supportsAudio = false,
            supportsGpu = false, // Uses NPU, not GPU
            requirements = ModelRequirements(minRamGB = 4, recommendedRamGB = 6),
            contextWindowSize = 0,
            modelFormat = "qnn_npu" // QNN SDK format for NPU acceleration
        ),
        LLMModel(
            name = "Absolute Reality (CPU)",
            description = "Absolute Reality SD1.5 model for CPU inference using MNN framework. Works on all Android devices without NPU requirements. Supports txt2img generation with flexible resolutions (128x128 to 512x512). Slower than NPU but compatible with all devices. ~1.2 GB download from HuggingFace.",
            url = "https://huggingface.co/xororz/sd-mnn/resolve/main/AbsoluteReality.zip",
            category = "image_generation",
            sizeBytes = 1288490188L, // ~1.2 GB
            source = "Stable Diffusion 1.5 (MNN via xororz)",
            supportsVision = false,
            supportsAudio = false,
            supportsGpu = false, // CPU only
            requirements = ModelRequirements(minRamGB = 2, recommendedRamGB = 4),
            contextWindowSize = 0,
            modelFormat = "mnn_cpu" // MNN framework for CPU inference
        )
        // Note: Gecko tokenizer removed - Gecko models have built-in tokenizers
    )

    /**
     * Current Status and Next Steps
     */
    fun getStatusMessage(): String {
        return """
        🎯 ON-DEVICE AI MODELS FOR ANDROID
        
        📱 READY FOR DIRECT DOWNLOAD:
        
        ✅ MULTIMODAL MODELS (Text + Vision + Audio):
        
        🔹 GEMMA-3N SERIES (Google):
        • Gemma-3n E2B (Vision+Audio+Text, 4k context) - 3.41GB
        • Gemma-3n E4B (Vision+Audio+Text, 4k context) - 4.58GB
        
        🔹 MINSTRAL-3 SERIES (MistralAI - Multimodal GGUF):
        • Ministral-3 3B Instruct (Vision enabled, 32k context)
        • Ministral-3 3B Reasoning (Thinking + Vision, 32k context)
        * Note: Vision support modules are now downloaded automatically with these models.
        
        ✅ TEXT MODELS:
        
        🔹 LFM-2.5 SERIES (LiquidAI - GGUF):
        • LFM-2.5 1.2B Instruct (128k context)
        • LFM-2.5 1.2B Thinking (128k context, Optimized for reasoning)
        
        🔹 GEMMA-3 SERIES (Google):
        • Gemma-3 1B (INT4/INT8, 2k-4k context)
        
        � LLAMA-3.2 SERIES (Meta):
        • Llama-3.2 1B/3B (INT8, 4k context)
        
        � PHI-4 SERIES (Microsoft):
        • Phi-4 Mini (INT8, 4k context) 
        
        � DOWNLOAD STATUS: Your app is ready for global model downloads via Nexa SDK integration!
        """.trimIndent()
    }
}
