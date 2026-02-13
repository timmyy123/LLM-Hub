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
        // App policy: expose GGUF NPU option only on 8 Gen 4 class devices.
        return getChipsetSuffix() == "8gen4"
    }

    /**
     * Normalize chipset suffix for SD QNN package naming.
     * Current hosted SD-QNN variants are keyed as 8gen1 and 8gen2, where 8gen2 is
     * also used for newer chips (8gen3/8gen4/8s/Elite class) for compatibility.
     */
    fun getSdQnnPackageSuffix(): String {
        return when (getChipsetSuffix()) {
            "8gen1" -> "8gen1"
            "8gen2", "8gen3", "8gen4" -> "8gen2"
            else -> "min"
        }
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

        // Llama-3.2 1B GGUF Models from bartowski
        LLMModel(
            name = "Llama-3.2 1B (IQ3_M)",
            description = "Llama 3.2 1B with IQ3_M quantization. Smallest size, great for low-memory devices. 128k context. (657MB)",
            url = "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-IQ3_M.gguf?download=true",
            category = "text",
            sizeBytes = 657000000L, // 657 MB
            source = "Meta via bartowski",
            supportsVision = false,
            supportsGpu = true,
            requirements = ModelRequirements(minRamGB = 2, recommendedRamGB = 3),
            contextWindowSize = 131072,
            modelFormat = "gguf"
        ),
        LLMModel(
            name = "Llama-3.2 1B (IQ4_XS)",
            description = "Llama 3.2 1B with IQ4_XS quantization. Optimized 4-bit. 128k context. (743MB)",
            url = "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-IQ4_XS.gguf?download=true",
            category = "text",
            sizeBytes = 743000000L, // 743 MB
            source = "Meta via bartowski",
            supportsVision = false,
            supportsGpu = true,
            requirements = ModelRequirements(minRamGB = 2, recommendedRamGB = 3),
            contextWindowSize = 131072,
            modelFormat = "gguf"
        ),
        LLMModel(
            name = "Llama-3.2 1B (Q3_K_L)",
            description = "Llama 3.2 1B with Q3_K_L quantization. Large 3-bit variant. 128k context. (733MB)",
            url = "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q3_K_L.gguf?download=true",
            category = "text",
            sizeBytes = 733000000L, // 733 MB
            source = "Meta via bartowski",
            supportsVision = false,
            supportsGpu = true,
            requirements = ModelRequirements(minRamGB = 2, recommendedRamGB = 3),
            contextWindowSize = 131072,
            modelFormat = "gguf"
        ),
        LLMModel(
            name = "Llama-3.2 1B (Q3_K_XL)",
            description = "Llama 3.2 1B with Q3_K_XL quantization. Extra-large 3-bit variant. 128k context. (796MB)",
            url = "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q3_K_XL.gguf?download=true",
            category = "text",
            sizeBytes = 796000000L, // 796 MB
            source = "Meta via bartowski",
            supportsVision = false,
            supportsGpu = true,
            requirements = ModelRequirements(minRamGB = 2, recommendedRamGB = 3),
            contextWindowSize = 131072,
            modelFormat = "gguf"
        ),
        LLMModel(
            name = "Llama-3.2 1B (Q4_0)",
            description = "Llama 3.2 1B with Q4_0 quantization. Standard 4-bit, good balance. 128k context. (773MB)",
            url = "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_0.gguf?download=true",
            category = "text",
            sizeBytes = 773000000L, // 773 MB
            source = "Meta via bartowski",
            supportsVision = false,
            supportsGpu = true,
            requirements = ModelRequirements(minRamGB = 2, recommendedRamGB = 3),
            contextWindowSize = 131072,
            modelFormat = "gguf"
        ),
        LLMModel(
            name = "Llama-3.2 1B (Q4_0_4_4)",
            description = "Llama 3.2 1B with Q4_0_4_4 quantization. 4-bit ARM-optimized. 128k context. (771MB)",
            url = "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_0_4_4.gguf?download=true",
            category = "text",
            sizeBytes = 771000000L, // 771 MB
            source = "Meta via bartowski",
            supportsVision = false,
            supportsGpu = true,
            requirements = ModelRequirements(minRamGB = 2, recommendedRamGB = 3),
            contextWindowSize = 131072,
            modelFormat = "gguf"
        ),
        LLMModel(
            name = "Llama-3.2 1B (Q4_0_4_8)",
            description = "Llama 3.2 1B with Q4_0_4_8 quantization. 4-bit ARM-optimized. 128k context. (771MB)",
            url = "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_0_4_8.gguf?download=true",
            category = "text",
            sizeBytes = 771000000L, // 771 MB
            source = "Meta via bartowski",
            supportsVision = false,
            supportsGpu = true,
            requirements = ModelRequirements(minRamGB = 2, recommendedRamGB = 3),
            contextWindowSize = 131072,
            modelFormat = "gguf"
        ),
        LLMModel(
            name = "Llama-3.2 1B (Q4_0_8_8)",
            description = "Llama 3.2 1B with Q4_0_8_8 quantization. 4-bit ARM-optimized. 128k context. (771MB)",
            url = "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_0_8_8.gguf?download=true",
            category = "text",
            sizeBytes = 771000000L, // 771 MB
            source = "Meta via bartowski",
            supportsVision = false,
            supportsGpu = true,
            requirements = ModelRequirements(minRamGB = 2, recommendedRamGB = 3),
            contextWindowSize = 131072,
            modelFormat = "gguf"
        ),
        LLMModel(
            name = "Llama-3.2 1B (Q4_K_L)",
            description = "Llama 3.2 1B with Q4_K_L quantization. Large K-quant 4-bit. 128k context. (871MB)",
            url = "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_L.gguf?download=true",
            category = "text",
            sizeBytes = 871000000L, // 871 MB
            source = "Meta via bartowski",
            supportsVision = false,
            supportsGpu = true,
            requirements = ModelRequirements(minRamGB = 2, recommendedRamGB = 3),
            contextWindowSize = 131072,
            modelFormat = "gguf"
        ),
        LLMModel(
            name = "Llama-3.2 1B (Q4_K_M)",
            description = "Llama 3.2 1B with Q4_K_M quantization. Medium K-quant 4-bit, recommended. 128k context. (808MB)",
            url = "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf?download=true",
            category = "text",
            sizeBytes = 808000000L, // 808 MB
            source = "Meta via bartowski",
            supportsVision = false,
            supportsGpu = true,
            requirements = ModelRequirements(minRamGB = 2, recommendedRamGB = 3),
            contextWindowSize = 131072,
            modelFormat = "gguf"
        ),
        LLMModel(
            name = "Llama-3.2 1B (Q4_K_S)",
            description = "Llama 3.2 1B with Q4_K_S quantization. Small K-quant 4-bit. 128k context. (776MB)",
            url = "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_S.gguf?download=true",
            category = "text",
            sizeBytes = 776000000L, // 776 MB
            source = "Meta via bartowski",
            supportsVision = false,
            supportsGpu = true,
            requirements = ModelRequirements(minRamGB = 2, recommendedRamGB = 3),
            contextWindowSize = 131072,
            modelFormat = "gguf"
        ),
        LLMModel(
            name = "Llama-3.2 1B (Q5_K_L)",
            description = "Llama 3.2 1B with Q5_K_L quantization. Large K-quant 5-bit. 128k context. (975MB)",
            url = "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q5_K_L.gguf?download=true",
            category = "text",
            sizeBytes = 975000000L, // 975 MB
            source = "Meta via bartowski",
            supportsVision = false,
            supportsGpu = true,
            requirements = ModelRequirements(minRamGB = 2, recommendedRamGB = 3),
            contextWindowSize = 131072,
            modelFormat = "gguf"
        ),
        LLMModel(
            name = "Llama-3.2 1B (Q5_K_M)",
            description = "Llama 3.2 1B with Q5_K_M quantization. Medium K-quant 5-bit, high quality. 128k context. (912MB)",
            url = "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q5_K_M.gguf?download=true",
            category = "text",
            sizeBytes = 912000000L, // 912 MB
            source = "Meta via bartowski",
            supportsVision = false,
            supportsGpu = true,
            requirements = ModelRequirements(minRamGB = 2, recommendedRamGB = 3),
            contextWindowSize = 131072,
            modelFormat = "gguf"
        ),
        LLMModel(
            name = "Llama-3.2 1B (Q5_K_S)",
            description = "Llama 3.2 1B with Q5_K_S quantization. Small K-quant 5-bit. 128k context. (893MB)",
            url = "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q5_K_S.gguf?download=true",
            category = "text",
            sizeBytes = 893000000L, // 893 MB
            source = "Meta via bartowski",
            supportsVision = false,
            supportsGpu = true,
            requirements = ModelRequirements(minRamGB = 2, recommendedRamGB = 3),
            contextWindowSize = 131072,
            modelFormat = "gguf"
        ),
        LLMModel(
            name = "Llama-3.2 1B (Q6_K)",
            description = "Llama 3.2 1B with Q6_K quantization. 6-bit, very high quality. 128k context. (1.02GB)",
            url = "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q6_K.gguf?download=true",
            category = "text",
            sizeBytes = 1020000000L, // 1.02 GB
            source = "Meta via bartowski",
            supportsVision = false,
            supportsGpu = true,
            requirements = ModelRequirements(minRamGB = 3, recommendedRamGB = 4),
            contextWindowSize = 131072,
            modelFormat = "gguf"
        ),
        LLMModel(
            name = "Llama-3.2 1B (Q6_K_L)",
            description = "Llama 3.2 1B with Q6_K_L quantization. Large 6-bit, highest quality. 128k context. (1.09GB)",
            url = "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q6_K_L.gguf?download=true",
            category = "text",
            sizeBytes = 1090000000L, // 1.09 GB
            source = "Meta via bartowski",
            supportsVision = false,
            supportsGpu = true,
            requirements = ModelRequirements(minRamGB = 3, recommendedRamGB = 4),
            contextWindowSize = 131072,
            modelFormat = "gguf"
        ),
        LLMModel(
            name = "Llama-3.2 1B (Q8_0)",
            description = "Llama 3.2 1B with Q8_0 quantization. 8-bit, near-original quality. 128k context. (1.32GB)",
            url = "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q8_0.gguf?download=true",
            category = "text",
            sizeBytes = 1320000000L, // 1.32 GB
            source = "Meta via bartowski",
            supportsVision = false,
            supportsGpu = true,
            requirements = ModelRequirements(minRamGB = 3, recommendedRamGB = 4),
            contextWindowSize = 131072,
            modelFormat = "gguf"
        ),
        LLMModel(
            name = "Llama-3.2 1B (f16)",
            description = "Llama 3.2 1B with f16 (full precision). Maximum quality, largest size. 128k context. (2.48GB)",
            url = "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-f16.gguf?download=true",
            category = "text",
            sizeBytes = 2480000000L, // 2.48 GB
            source = "Meta via bartowski",
            supportsVision = false,
            supportsGpu = true,
            requirements = ModelRequirements(minRamGB = 4, recommendedRamGB = 6),
            contextWindowSize = 131072,
            modelFormat = "gguf"
        ),

        // Llama-3.2 3B GGUF Models from bartowski
        LLMModel(
            name = "Llama-3.2 3B (IQ3_M)",
            description = "Llama 3.2 3B with IQ3_M quantization. Smallest size, great for low-memory devices. 128k context. (1.6GB)",
            url = "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-IQ3_M.gguf?download=true",
            category = "text",
            sizeBytes = 1600000000L, // 1.6 GB
            source = "Meta via bartowski",
            supportsVision = false,
            supportsGpu = true,
            requirements = ModelRequirements(minRamGB = 3, recommendedRamGB = 4),
            contextWindowSize = 131072,
            modelFormat = "gguf"
        ),
        LLMModel(
            name = "Llama-3.2 3B (IQ4_XS)",
            description = "Llama 3.2 3B with IQ4_XS quantization. Optimized 4-bit. 128k context. (1.83GB)",
            url = "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-IQ4_XS.gguf?download=true",
            category = "text",
            sizeBytes = 1830000000L, // 1.83 GB
            source = "Meta via bartowski",
            supportsVision = false,
            supportsGpu = true,
            requirements = ModelRequirements(minRamGB = 3, recommendedRamGB = 4),
            contextWindowSize = 131072,
            modelFormat = "gguf"
        ),
        LLMModel(
            name = "Llama-3.2 3B (Q3_K_L)",
            description = "Llama 3.2 3B with Q3_K_L quantization. Large 3-bit variant. 128k context. (1.82GB)",
            url = "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q3_K_L.gguf?download=true",
            category = "text",
            sizeBytes = 1820000000L, // 1.82 GB
            source = "Meta via bartowski",
            supportsVision = false,
            supportsGpu = true,
            requirements = ModelRequirements(minRamGB = 3, recommendedRamGB = 4),
            contextWindowSize = 131072,
            modelFormat = "gguf"
        ),
        LLMModel(
            name = "Llama-3.2 3B (Q3_K_XL)",
            description = "Llama 3.2 3B with Q3_K_XL quantization. Extra-large 3-bit variant. 128k context. (1.91GB)",
            url = "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q3_K_XL.gguf?download=true",
            category = "text",
            sizeBytes = 1910000000L, // 1.91 GB
            source = "Meta via bartowski",
            supportsVision = false,
            supportsGpu = true,
            requirements = ModelRequirements(minRamGB = 3, recommendedRamGB = 4),
            contextWindowSize = 131072,
            modelFormat = "gguf"
        ),
        LLMModel(
            name = "Llama-3.2 3B (Q4_0)",
            description = "Llama 3.2 3B with Q4_0 quantization. Standard 4-bit, good balance. 128k context. (1.92GB)",
            url = "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_0.gguf?download=true",
            category = "text",
            sizeBytes = 1920000000L, // 1.92 GB
            source = "Meta via bartowski",
            supportsVision = false,
            supportsGpu = true,
            requirements = ModelRequirements(minRamGB = 3, recommendedRamGB = 4),
            contextWindowSize = 131072,
            modelFormat = "gguf"
        ),
        LLMModel(
            name = "Llama-3.2 3B (Q4_0_4_4)",
            description = "Llama 3.2 3B with Q4_0_4_4 quantization. 4-bit ARM-optimized. 128k context. (1.92GB)",
            url = "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_0_4_4.gguf?download=true",
            category = "text",
            sizeBytes = 1920000000L, // 1.92 GB
            source = "Meta via bartowski",
            supportsVision = false,
            supportsGpu = true,
            requirements = ModelRequirements(minRamGB = 3, recommendedRamGB = 4),
            contextWindowSize = 131072,
            modelFormat = "gguf"
        ),
        LLMModel(
            name = "Llama-3.2 3B (Q4_0_4_8)",
            description = "Llama 3.2 3B with Q4_0_4_8 quantization. 4-bit ARM-optimized. 128k context. (1.92GB)",
            url = "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_0_4_8.gguf?download=true",
            category = "text",
            sizeBytes = 1920000000L, // 1.92 GB
            source = "Meta via bartowski",
            supportsVision = false,
            supportsGpu = true,
            requirements = ModelRequirements(minRamGB = 3, recommendedRamGB = 4),
            contextWindowSize = 131072,
            modelFormat = "gguf"
        ),
        LLMModel(
            name = "Llama-3.2 3B (Q4_0_8_8)",
            description = "Llama 3.2 3B with Q4_0_8_8 quantization. 4-bit ARM-optimized. 128k context. (1.92GB)",
            url = "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_0_8_8.gguf?download=true",
            category = "text",
            sizeBytes = 1920000000L, // 1.92 GB
            source = "Meta via bartowski",
            supportsVision = false,
            supportsGpu = true,
            requirements = ModelRequirements(minRamGB = 3, recommendedRamGB = 4),
            contextWindowSize = 131072,
            modelFormat = "gguf"
        ),
        LLMModel(
            name = "Llama-3.2 3B (Q4_K_L)",
            description = "Llama 3.2 3B with Q4_K_L quantization. Large K-quant 4-bit. 128k context. (2.11GB)",
            url = "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_L.gguf?download=true",
            category = "text",
            sizeBytes = 2110000000L, // 2.11 GB
            source = "Meta via bartowski",
            supportsVision = false,
            supportsGpu = true,
            requirements = ModelRequirements(minRamGB = 3, recommendedRamGB = 5),
            contextWindowSize = 131072,
            modelFormat = "gguf"
        ),
        LLMModel(
            name = "Llama-3.2 3B (Q4_K_M)",
            description = "Llama 3.2 3B with Q4_K_M quantization. Medium K-quant 4-bit, recommended. 128k context. (2.02GB)",
            url = "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf?download=true",
            category = "text",
            sizeBytes = 2020000000L, // 2.02 GB
            source = "Meta via bartowski",
            supportsVision = false,
            supportsGpu = true,
            requirements = ModelRequirements(minRamGB = 3, recommendedRamGB = 5),
            contextWindowSize = 131072,
            modelFormat = "gguf"
        ),
        LLMModel(
            name = "Llama-3.2 3B (Q4_K_S)",
            description = "Llama 3.2 3B with Q4_K_S quantization. Small K-quant 4-bit. 128k context. (1.93GB)",
            url = "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_S.gguf?download=true",
            category = "text",
            sizeBytes = 1930000000L, // 1.93 GB
            source = "Meta via bartowski",
            supportsVision = false,
            supportsGpu = true,
            requirements = ModelRequirements(minRamGB = 3, recommendedRamGB = 4),
            contextWindowSize = 131072,
            modelFormat = "gguf"
        ),
        LLMModel(
            name = "Llama-3.2 3B (Q5_K_L)",
            description = "Llama 3.2 3B with Q5_K_L quantization. Large K-quant 5-bit. 128k context. (2.42GB)",
            url = "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q5_K_L.gguf?download=true",
            category = "text",
            sizeBytes = 2420000000L, // 2.42 GB
            source = "Meta via bartowski",
            supportsVision = false,
            supportsGpu = true,
            requirements = ModelRequirements(minRamGB = 4, recommendedRamGB = 5),
            contextWindowSize = 131072,
            modelFormat = "gguf"
        ),
        LLMModel(
            name = "Llama-3.2 3B (Q5_K_M)",
            description = "Llama 3.2 3B with Q5_K_M quantization. Medium K-quant 5-bit, high quality. 128k context. (2.32GB)",
            url = "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q5_K_M.gguf?download=true",
            category = "text",
            sizeBytes = 2320000000L, // 2.32 GB
            source = "Meta via bartowski",
            supportsVision = false,
            supportsGpu = true,
            requirements = ModelRequirements(minRamGB = 4, recommendedRamGB = 5),
            contextWindowSize = 131072,
            modelFormat = "gguf"
        ),
        LLMModel(
            name = "Llama-3.2 3B (Q5_K_S)",
            description = "Llama 3.2 3B with Q5_K_S quantization. Small K-quant 5-bit. 128k context. (2.27GB)",
            url = "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q5_K_S.gguf?download=true",
            category = "text",
            sizeBytes = 2270000000L, // 2.27 GB
            source = "Meta via bartowski",
            supportsVision = false,
            supportsGpu = true,
            requirements = ModelRequirements(minRamGB = 4, recommendedRamGB = 5),
            contextWindowSize = 131072,
            modelFormat = "gguf"
        ),
        LLMModel(
            name = "Llama-3.2 3B (Q6_K)",
            description = "Llama 3.2 3B with Q6_K quantization. 6-bit, very high quality. 128k context. (2.64GB)",
            url = "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q6_K.gguf?download=true",
            category = "text",
            sizeBytes = 2640000000L, // 2.64 GB
            source = "Meta via bartowski",
            supportsVision = false,
            supportsGpu = true,
            requirements = ModelRequirements(minRamGB = 4, recommendedRamGB = 6),
            contextWindowSize = 131072,
            modelFormat = "gguf"
        ),
        LLMModel(
            name = "Llama-3.2 3B (Q6_K_L)",
            description = "Llama 3.2 3B with Q6_K_L quantization. Large 6-bit, highest quality. 128k context. (2.74GB)",
            url = "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q6_K_L.gguf?download=true",
            category = "text",
            sizeBytes = 2740000000L, // 2.74 GB
            source = "Meta via bartowski",
            supportsVision = false,
            supportsGpu = true,
            requirements = ModelRequirements(minRamGB = 4, recommendedRamGB = 6),
            contextWindowSize = 131072,
            modelFormat = "gguf"
        ),
        LLMModel(
            name = "Llama-3.2 3B (Q8_0)",
            description = "Llama 3.2 3B with Q8_0 quantization. 8-bit, near-original quality. 128k context. (3.42GB)",
            url = "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q8_0.gguf?download=true",
            category = "text",
            sizeBytes = 3420000000L, // 3.42 GB
            source = "Meta via bartowski",
            supportsVision = false,
            supportsGpu = true,
            requirements = ModelRequirements(minRamGB = 5, recommendedRamGB = 7),
            contextWindowSize = 131072,
            modelFormat = "gguf"
        ),
        LLMModel(
            name = "Llama-3.2 3B (f16)",
            description = "Llama 3.2 3B with f16 (full precision). Maximum quality, largest size. 128k context. (6.43GB)",
            url = "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-f16.gguf?download=true",
            category = "text",
            sizeBytes = 6430000000L, // 6.43 GB
            source = "Meta via bartowski",
            supportsVision = false,
            supportsGpu = true,
            requirements = ModelRequirements(minRamGB = 8, recommendedRamGB = 10),
            contextWindowSize = 131072,
            modelFormat = "gguf"
        ),

        // Granite 4.0 H-Tiny 7B models (IBM)
        LLMModel(
            name = "Granite 4.0 H-Tiny (Q2_K)",
            description = "IBM Granite 4.0 H-Tiny with Q2_K quantization. Smallest size. 128k context. (2.59GB)",
            url = "https://huggingface.co/ibm-granite/granite-4.0-h-tiny-GGUF/resolve/main/granite-4.0-h-tiny-Q2_K.gguf?download=true",
            category = "text",
            sizeBytes = 2590000000L, // 2.59 GB
            source = "IBM Granite",
            supportsVision = false,
            supportsGpu = true,
            requirements = ModelRequirements(minRamGB = 3, recommendedRamGB = 4),
            contextWindowSize = 131072,
            modelFormat = "gguf"
        ),
        LLMModel(
            name = "Granite 4.0 H-Tiny (Q3_K_S)",
            description = "IBM Granite 4.0 H-Tiny with Q3_K_S quantization. Balanced size. 128k context. (3.06GB)",
            url = "https://huggingface.co/ibm-granite/granite-4.0-h-tiny-GGUF/resolve/main/granite-4.0-h-tiny-Q3_K_S.gguf?download=true",
            category = "text",
            sizeBytes = 3060000000L, // 3.06 GB
            source = "IBM Granite",
            supportsVision = false,
            supportsGpu = true,
            requirements = ModelRequirements(minRamGB = 4, recommendedRamGB = 5),
            contextWindowSize = 131072,
            modelFormat = "gguf"
        ),
        LLMModel(
            name = "Granite 4.0 H-Tiny (Q3_K_M)",
            description = "IBM Granite 4.0 H-Tiny with Q3_K_M quantization. Good quality. 128k context. (3.35GB)",
            url = "https://huggingface.co/ibm-granite/granite-4.0-h-tiny-GGUF/resolve/main/granite-4.0-h-tiny-Q3_K_M.gguf?download=true",
            category = "text",
            sizeBytes = 3350000000L, // 3.35 GB
            source = "IBM Granite",
            supportsVision = false,
            supportsGpu = true,
            requirements = ModelRequirements(minRamGB = 4, recommendedRamGB = 5),
            contextWindowSize = 131072,
            modelFormat = "gguf"
        ),
        LLMModel(
            name = "Granite 4.0 H-Tiny (Q3_K_L)",
            description = "IBM Granite 4.0 H-Tiny with Q3_K_L quantization. Better quality. 128k context. (3.6GB)",
            url = "https://huggingface.co/ibm-granite/granite-4.0-h-tiny-GGUF/resolve/main/granite-4.0-h-tiny-Q3_K_L.gguf?download=true",
            category = "text",
            sizeBytes = 3600000000L, // 3.6 GB
            source = "IBM Granite",
            supportsVision = false,
            supportsGpu = true,
            requirements = ModelRequirements(minRamGB = 4, recommendedRamGB = 5),
            contextWindowSize = 131072,
            modelFormat = "gguf"
        ),
        LLMModel(
            name = "Granite 4.0 H-Tiny (Q4_0)",
            description = "IBM Granite 4.0 H-Tiny with Q4_0 quantization. Good balance. 128k context. (3.96GB)",
            url = "https://huggingface.co/ibm-granite/granite-4.0-h-tiny-GGUF/resolve/main/granite-4.0-h-tiny-Q4_0.gguf?download=true",
            category = "text",
            sizeBytes = 3960000000L, // 3.96 GB
            source = "IBM Granite",
            supportsVision = false,
            supportsGpu = true,
            requirements = ModelRequirements(minRamGB = 5, recommendedRamGB = 6),
            contextWindowSize = 131072,
            modelFormat = "gguf"
        ),
        LLMModel(
            name = "Granite 4.0 H-Tiny (Q4_K_S)",
            description = "IBM Granite 4.0 H-Tiny with Q4_K_S quantization. High quality. 128k context. (4GB)",
            url = "https://huggingface.co/ibm-granite/granite-4.0-h-tiny-GGUF/resolve/main/granite-4.0-h-tiny-Q4_K_S.gguf?download=true",
            category = "text",
            sizeBytes = 4000000000L, // 4 GB
            source = "IBM Granite",
            supportsVision = false,
            supportsGpu = true,
            requirements = ModelRequirements(minRamGB = 5, recommendedRamGB = 6),
            contextWindowSize = 131072,
            modelFormat = "gguf"
        ),
        LLMModel(
            name = "Granite 4.0 H-Tiny (Q4_K_M)",
            description = "IBM Granite 4.0 H-Tiny with Q4_K_M quantization. Very high quality. 128k context. (4.23GB)",
            url = "https://huggingface.co/ibm-granite/granite-4.0-h-tiny-GGUF/resolve/main/granite-4.0-h-tiny-Q4_K_M.gguf?download=true",
            category = "text",
            sizeBytes = 4230000000L, // 4.23 GB
            source = "IBM Granite",
            supportsVision = false,
            supportsGpu = true,
            requirements = ModelRequirements(minRamGB = 5, recommendedRamGB = 6),
            contextWindowSize = 131072,
            modelFormat = "gguf"
        ),
        LLMModel(
            name = "Granite 4.0 H-Tiny (Q4_1)",
            description = "IBM Granite 4.0 H-Tiny with Q4_1 quantization. Enhanced quality. 128k context. (4.39GB)",
            url = "https://huggingface.co/ibm-granite/granite-4.0-h-tiny-GGUF/resolve/main/granite-4.0-h-tiny-Q4_1.gguf?download=true",
            category = "text",
            sizeBytes = 4390000000L, // 4.39 GB
            source = "IBM Granite",
            supportsVision = false,
            supportsGpu = true,
            requirements = ModelRequirements(minRamGB = 5, recommendedRamGB = 6),
            contextWindowSize = 131072,
            modelFormat = "gguf"
        ),
        LLMModel(
            name = "Granite 4.0 H-Tiny (Q5_K_S)",
            description = "IBM Granite 4.0 H-Tiny with Q5_K_S quantization. Excellent quality. 128k context. (4.81GB)",
            url = "https://huggingface.co/ibm-granite/granite-4.0-h-tiny-GGUF/resolve/main/granite-4.0-h-tiny-Q5_K_S.gguf?download=true",
            category = "text",
            sizeBytes = 4810000000L, // 4.81 GB
            source = "IBM Granite",
            supportsVision = false,
            supportsGpu = true,
            requirements = ModelRequirements(minRamGB = 6, recommendedRamGB = 7),
            contextWindowSize = 131072,
            modelFormat = "gguf"
        ),
        LLMModel(
            name = "Granite 4.0 H-Tiny (Q5_0)",
            description = "IBM Granite 4.0 H-Tiny with Q5_0 quantization. Near-lossless. 128k context. (4.81GB)",
            url = "https://huggingface.co/ibm-granite/granite-4.0-h-tiny-GGUF/resolve/main/granite-4.0-h-tiny-Q5_0.gguf?download=true",
            category = "text",
            sizeBytes = 4810000000L, // 4.81 GB
            source = "IBM Granite",
            supportsVision = false,
            supportsGpu = true,
            requirements = ModelRequirements(minRamGB = 6, recommendedRamGB = 7),
            contextWindowSize = 131072,
            modelFormat = "gguf"
        ),
        LLMModel(
            name = "Granite 4.0 H-Tiny (Q5_K_M)",
            description = "IBM Granite 4.0 H-Tiny with Q5_K_M quantization. Superior quality. 128k context. (4.95GB)",
            url = "https://huggingface.co/ibm-granite/granite-4.0-h-tiny-GGUF/resolve/main/granite-4.0-h-tiny-Q5_K_M.gguf?download=true",
            category = "text",
            sizeBytes = 4950000000L, // 4.95 GB
            source = "IBM Granite",
            supportsVision = false,
            supportsGpu = true,
            requirements = ModelRequirements(minRamGB = 6, recommendedRamGB = 7),
            contextWindowSize = 131072,
            modelFormat = "gguf"
        ),
        LLMModel(
            name = "Granite 4.0 H-Tiny (Q5_1)",
            description = "IBM Granite 4.0 H-Tiny with Q5_1 quantization. Premium quality. 128k context. (5.23GB)",
            url = "https://huggingface.co/ibm-granite/granite-4.0-h-tiny-GGUF/resolve/main/granite-4.0-h-tiny-Q5_1.gguf?download=true",
            category = "text",
            sizeBytes = 5230000000L, // 5.23 GB
            source = "IBM Granite",
            supportsVision = false,
            supportsGpu = true,
            requirements = ModelRequirements(minRamGB = 6, recommendedRamGB = 7),
            contextWindowSize = 131072,
            modelFormat = "gguf"
        ),
        LLMModel(
            name = "Granite 4.0 H-Tiny (Q6_K)",
            description = "IBM Granite 4.0 H-Tiny with Q6_K quantization. Outstanding quality. 128k context. (5.71GB)",
            url = "https://huggingface.co/ibm-granite/granite-4.0-h-tiny-GGUF/resolve/main/granite-4.0-h-tiny-Q6_K.gguf?download=true",
            category = "text",
            sizeBytes = 5710000000L, // 5.71 GB
            source = "IBM Granite",
            supportsVision = false,
            supportsGpu = true,
            requirements = ModelRequirements(minRamGB = 7, recommendedRamGB = 8),
            contextWindowSize = 131072,
            modelFormat = "gguf"
        ),
        LLMModel(
            name = "Granite 4.0 H-Tiny (Q8_0)",
            description = "IBM Granite 4.0 H-Tiny with Q8_0 quantization. Ultimate quality. 128k context. (7.39GB)",
            url = "https://huggingface.co/ibm-granite/granite-4.0-h-tiny-GGUF/resolve/main/granite-4.0-h-tiny-Q8_0.gguf?download=true",
            category = "text",
            sizeBytes = 7390000000L, // 7.39 GB
            source = "IBM Granite",
            supportsVision = false,
            supportsGpu = true,
            requirements = ModelRequirements(minRamGB = 8, recommendedRamGB = 10),
            contextWindowSize = 131072,
            modelFormat = "gguf"
        ),
        LLMModel(
            name = "Granite 4.0 H-Tiny (f16)",
            description = "IBM Granite 4.0 H-Tiny with f16 (full precision). Maximum quality, largest size. 128k context. (13.9GB)",
            url = "https://huggingface.co/ibm-granite/granite-4.0-h-tiny-GGUF/resolve/main/granite-4.0-h-tiny-f16.gguf?download=true",
            category = "text",
            sizeBytes = 13900000000L, // 13.9 GB
            source = "IBM Granite",
            supportsVision = false,
            supportsGpu = true,
            requirements = ModelRequirements(minRamGB = 14, recommendedRamGB = 16),
            contextWindowSize = 131072,
            modelFormat = "gguf"
        ),

        // Granite 4.0 H-Small 32B models (IBM)
        LLMModel(
            name = "Granite 4.0 H-Small (Q2_K)",
            description = "IBM Granite 4.0 H-Small with Q2_K quantization. Smallest size. 128k context. (11.8GB)",
            url = "https://huggingface.co/ibm-granite/granite-4.0-h-small-GGUF/resolve/main/granite-4.0-h-small-Q2_K.gguf?download=true",
            category = "text",
            sizeBytes = 11800000000L, // 11.8 GB
            source = "IBM Granite",
            supportsVision = false,
            supportsGpu = true,
            requirements = ModelRequirements(minRamGB = 12, recommendedRamGB = 14),
            contextWindowSize = 131072,
            modelFormat = "gguf"
        ),
        LLMModel(
            name = "Granite 4.0 H-Small (Q3_K_S)",
            description = "IBM Granite 4.0 H-Small with Q3_K_S quantization. Balanced size. 128k context. (14.1GB)",
            url = "https://huggingface.co/ibm-granite/granite-4.0-h-small-GGUF/resolve/main/granite-4.0-h-small-Q3_K_S.gguf?download=true",
            category = "text",
            sizeBytes = 14100000000L, // 14.1 GB
            source = "IBM Granite",
            supportsVision = false,
            supportsGpu = true,
            requirements = ModelRequirements(minRamGB = 14, recommendedRamGB = 16),
            contextWindowSize = 131072,
            modelFormat = "gguf"
        ),
        LLMModel(
            name = "Granite 4.0 H-Small (Q3_K_M)",
            description = "IBM Granite 4.0 H-Small with Q3_K_M quantization. Good quality. 128k context. (15.4GB)",
            url = "https://huggingface.co/ibm-granite/granite-4.0-h-small-GGUF/resolve/main/granite-4.0-h-small-Q3_K_M.gguf?download=true",
            category = "text",
            sizeBytes = 15400000000L, // 15.4 GB
            source = "IBM Granite",
            supportsVision = false,
            supportsGpu = true,
            requirements = ModelRequirements(minRamGB = 15, recommendedRamGB = 18),
            contextWindowSize = 131072,
            modelFormat = "gguf"
        ),
        LLMModel(
            name = "Granite 4.0 H-Small (Q3_K_L)",
            description = "IBM Granite 4.0 H-Small with Q3_K_L quantization. Better quality. 128k context. (16.5GB)",
            url = "https://huggingface.co/ibm-granite/granite-4.0-h-small-GGUF/resolve/main/granite-4.0-h-small-Q3_K_L.gguf?download=true",
            category = "text",
            sizeBytes = 16500000000L, // 16.5 GB
            source = "IBM Granite",
            supportsVision = false,
            supportsGpu = true,
            requirements = ModelRequirements(minRamGB = 16, recommendedRamGB = 20),
            contextWindowSize = 131072,
            modelFormat = "gguf"
        ),
        LLMModel(
            name = "Granite 4.0 H-Small (Q4_0)",
            description = "IBM Granite 4.0 H-Small with Q4_0 quantization. Good balance. 128k context. (18.3GB)",
            url = "https://huggingface.co/ibm-granite/granite-4.0-h-small-GGUF/resolve/main/granite-4.0-h-small-Q4_0.gguf?download=true",
            category = "text",
            sizeBytes = 18300000000L, // 18.3 GB
            source = "IBM Granite",
            supportsVision = false,
            supportsGpu = true,
            requirements = ModelRequirements(minRamGB = 18, recommendedRamGB = 22),
            contextWindowSize = 131072,
            modelFormat = "gguf"
        ),
        LLMModel(
            name = "Granite 4.0 H-Small (Q4_K_S)",
            description = "IBM Granite 4.0 H-Small with Q4_K_S quantization. High quality. 128k context. (18.4GB)",
            url = "https://huggingface.co/ibm-granite/granite-4.0-h-small-GGUF/resolve/main/granite-4.0-h-small-Q4_K_S.gguf?download=true",
            category = "text",
            sizeBytes = 18400000000L, // 18.4 GB
            source = "IBM Granite",
            supportsVision = false,
            supportsGpu = true,
            requirements = ModelRequirements(minRamGB = 18, recommendedRamGB = 22),
            contextWindowSize = 131072,
            modelFormat = "gguf"
        ),
        LLMModel(
            name = "Granite 4.0 H-Small (Q4_K_M)",
            description = "IBM Granite 4.0 H-Small with Q4_K_M quantization. Very high quality. 128k context. (19.5GB)",
            url = "https://huggingface.co/ibm-granite/granite-4.0-h-small-GGUF/resolve/main/granite-4.0-h-small-Q4_K_M.gguf?download=true",
            category = "text",
            sizeBytes = 19500000000L, // 19.5 GB
            source = "IBM Granite",
            supportsVision = false,
            supportsGpu = true,
            requirements = ModelRequirements(minRamGB = 19, recommendedRamGB = 24),
            contextWindowSize = 131072,
            modelFormat = "gguf"
        ),
        LLMModel(
            name = "Granite 4.0 H-Small (Q4_1)",
            description = "IBM Granite 4.0 H-Small with Q4_1 quantization. Enhanced quality. 128k context. (20.3GB)",
            url = "https://huggingface.co/ibm-granite/granite-4.0-h-small-GGUF/resolve/main/granite-4.0-h-small-Q4_1.gguf?download=true",
            category = "text",
            sizeBytes = 20300000000L, // 20.3 GB
            source = "IBM Granite",
            supportsVision = false,
            supportsGpu = true,
            requirements = ModelRequirements(minRamGB = 20, recommendedRamGB = 24),
            contextWindowSize = 131072,
            modelFormat = "gguf"
        ),
        LLMModel(
            name = "Granite 4.0 H-Small (Q5_0)",
            description = "IBM Granite 4.0 H-Small with Q5_0 quantization. Near-lossless. 128k context. (22.2GB)",
            url = "https://huggingface.co/ibm-granite/granite-4.0-h-small-GGUF/resolve/main/granite-4.0-h-small-Q5_0.gguf?download=true",
            category = "text",
            sizeBytes = 22200000000L, // 22.2 GB
            source = "IBM Granite",
            supportsVision = false,
            supportsGpu = true,
            requirements = ModelRequirements(minRamGB = 22, recommendedRamGB = 26),
            contextWindowSize = 131072,
            modelFormat = "gguf"
        ),
        LLMModel(
            name = "Granite 4.0 H-Small (Q5_K_S)",
            description = "IBM Granite 4.0 H-Small with Q5_K_S quantization. Excellent quality. 128k context. (22.2GB)",
            url = "https://huggingface.co/ibm-granite/granite-4.0-h-small-GGUF/resolve/main/granite-4.0-h-small-Q5_K_S.gguf?download=true",
            category = "text",
            sizeBytes = 22200000000L, // 22.2 GB
            source = "IBM Granite",
            supportsVision = false,
            supportsGpu = true,
            requirements = ModelRequirements(minRamGB = 22, recommendedRamGB = 26),
            contextWindowSize = 131072,
            modelFormat = "gguf"
        ),
        LLMModel(
            name = "Granite 4.0 H-Small (Q5_K_M)",
            description = "IBM Granite 4.0 H-Small with Q5_K_M quantization. Superior quality. 128k context. (22.9GB)",
            url = "https://huggingface.co/ibm-granite/granite-4.0-h-small-GGUF/resolve/main/granite-4.0-h-small-Q5_K_M.gguf?download=true",
            category = "text",
            sizeBytes = 22900000000L, // 22.9 GB
            source = "IBM Granite",
            supportsVision = false,
            supportsGpu = true,
            requirements = ModelRequirements(minRamGB = 23, recommendedRamGB = 26),
            contextWindowSize = 131072,
            modelFormat = "gguf"
        ),
        LLMModel(
            name = "Granite 4.0 H-Small (Q5_1)",
            description = "IBM Granite 4.0 H-Small with Q5_1 quantization. Premium quality. 128k context. (24.2GB)",
            url = "https://huggingface.co/ibm-granite/granite-4.0-h-small-GGUF/resolve/main/granite-4.0-h-small-Q5_1.gguf?download=true",
            category = "text",
            sizeBytes = 24200000000L, // 24.2 GB
            source = "IBM Granite",
            supportsVision = false,
            supportsGpu = true,
            requirements = ModelRequirements(minRamGB = 24, recommendedRamGB = 28),
            contextWindowSize = 131072,
            modelFormat = "gguf"
        ),
        LLMModel(
            name = "Granite 4.0 H-Small (Q6_K)",
            description = "IBM Granite 4.0 H-Small with Q6_K quantization. Outstanding quality. 128k context. (26.5GB)",
            url = "https://huggingface.co/ibm-granite/granite-4.0-h-small-GGUF/resolve/main/granite-4.0-h-small-Q6_K.gguf?download=true",
            category = "text",
            sizeBytes = 26500000000L, // 26.5 GB
            source = "IBM Granite",
            supportsVision = false,
            supportsGpu = true,
            requirements = ModelRequirements(minRamGB = 26, recommendedRamGB = 30),
            contextWindowSize = 131072,
            modelFormat = "gguf"
        ),
        LLMModel(
            name = "Granite 4.0 H-Small (Q8_0)",
            description = "IBM Granite 4.0 H-Small with Q8_0 quantization. Ultimate quality. 128k context. (34.3GB)",
            url = "https://huggingface.co/ibm-granite/granite-4.0-h-small-GGUF/resolve/main/granite-4.0-h-small-Q8_0.gguf?download=true",
            category = "text",
            sizeBytes = 34300000000L, // 34.3 GB
            source = "IBM Granite",
            supportsVision = false,
            supportsGpu = true,
            requirements = ModelRequirements(minRamGB = 34, recommendedRamGB = 40),
            contextWindowSize = 131072,
            modelFormat = "gguf"
        ),
        LLMModel(
            name = "Granite 4.0 H-Small (f16)",
            description = "IBM Granite 4.0 H-Small with f16 (full precision). Maximum quality, largest size. 128k context. (64.4GB)",
            url = "https://huggingface.co/ibm-granite/granite-4.0-h-small-GGUF/resolve/main/granite-4.0-h-small-f16.gguf?download=true",
            category = "text",
            sizeBytes = 64400000000L, // 64.4 GB
            source = "IBM Granite",
            supportsVision = false,
            supportsGpu = true,
            requirements = ModelRequirements(minRamGB = 64, recommendedRamGB = 80),
            contextWindowSize = 131072,
            modelFormat = "gguf"
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

        // --- Added Gemma-3 GGUF multimodal models (4B & 12B) + vision projectors ---
        LLMModel(
            name = "Gemma-3 4B (Q4_0, GGUF)",
            description = "Google Gemma-3 4B quantized GGUF (Q4_0). Supports text + vision — download the Vision Projector (mmproj) to enable image input.",
            url = "https://huggingface.co/google/gemma-3-4b-it-qat-q4_0-gguf/resolve/main/gemma-3-4b-it-q4_0.gguf?download=true",
            category = "multimodal",
            sizeBytes = 3155051328L, // 3,155,051,328 bytes (~3008.89 MB)
            source = "Google",
            supportsVision = true,
            supportsAudio = false,
            supportsGpu = true,
            requirements = ModelRequirements(minRamGB = 6, recommendedRamGB = 8),
            contextWindowSize = 4096,
            modelFormat = "gguf"
        ),
        LLMModel(
            name = "Gemma-3 4B (Vision Projector, BF16)",
            description = "Vision Projector (mmproj) required to enable image input for Gemma-3 4B. BF16 variant for accurate visual encodings.",
            url = "https://huggingface.co/google/gemma-3-4b-it-qat-q4_0-gguf/resolve/main/mmproj-model-f16-4B.gguf?download=true",
            category = "multimodal",
            sizeBytes = 851251104L, // 851,251,104 bytes (~811.82 MB)
            source = "Google",
            supportsVision = true,
            supportsGpu = true,
            requirements = ModelRequirements(minRamGB = 1, recommendedRamGB = 2),
            contextWindowSize = 0,
            modelFormat = "gguf"
        ),
        LLMModel(
            name = "Gemma-3 12B (Q4_0, GGUF)",
            description = "Google Gemma-3 12B quantized GGUF (Q4_0). Supports text + vision — download the Vision Projector (mmproj) to enable image input.",
            url = "https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-gguf/resolve/main/gemma-3-12b-it-q4_0.gguf?download=true",
            category = "multimodal",
            sizeBytes = 8074473920L, // 8,074,473,920 bytes (~7700.42 MB)
            source = "Google",
            supportsVision = true,
            supportsAudio = false,
            supportsGpu = true,
            requirements = ModelRequirements(minRamGB = 12, recommendedRamGB = 16),
            contextWindowSize = 4096,
            modelFormat = "gguf"
        ),
        LLMModel(
            name = "Gemma-3 12B (Vision Projector, BF16)",
            description = "Vision Projector (mmproj) required to enable image input for Gemma-3 12B. BF16 variant for accurate visual encodings.",
            url = "https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-gguf/resolve/main/mmproj-model-f16-12B.gguf?download=true",
            category = "multimodal",
            sizeBytes = 854200224L, // 854,200,224 bytes (~814.63 MB)
            source = "Google",
            supportsVision = true,
            supportsGpu = true,
            requirements = ModelRequirements(minRamGB = 1, recommendedRamGB = 2),
            contextWindowSize = 0,
            modelFormat = "gguf"
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
            name = "Absolute Reality (NPU - ${DeviceInfo.getSdQnnPackageSuffix()})",
            description = "Absolute Reality SD1.5 model optimized for Qualcomm NPU acceleration using QNN SDK. Supports txt2img generation at 512x512 resolution. Requires Snapdragon 8 Gen 1 or newer with Hexagon NPU. Device detected: ${DeviceInfo.getDeviceSoc()}. ~1.06 GB download from HuggingFace.",
            url = "https://huggingface.co/xororz/sd-qnn/resolve/main/AbsoluteReality_qnn2.28_${DeviceInfo.getSdQnnPackageSuffix()}.zip",
            category = "image_generation",
            sizeBytes = when(DeviceInfo.getSdQnnPackageSuffix()) {
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
