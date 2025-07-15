package com.example.llmhub.data

object ModelData {
    val models = listOf(
        // Gemma-3 1B Models
        LLMModel(
            name = "Gemma-3 1B (INT4, 2k)",
            description = "Google Gemma-3 1B with INT4 quantization and a 2k context window. Optimized for mobile devices. Ready to download from HuggingFace (555MB)",
            url = "https://huggingface.co/litert-community/Gemma3-1B-IT/resolve/main/Gemma3-1B-IT_multi-prefill-seq_q4_ekv2048.task?download=true",
            category = "text",
            sizeBytes = 555000000L, // 555 MB
            source = "Google via LiteRT Community",
            supportsVision = false,
            requirements = ModelRequirements(minRamGB = 2, recommendedRamGB = 3),
            contextWindowSize = 2048,
            modelFormat = "task"
        ),
        LLMModel(
            name = "Gemma-3 1B (INT8, 1.2k)",
            description = "Higher quality INT8 version of Gemma-3 1B with a 1.2k context window. Ready to download from HuggingFace (1.05GB)",
            url = "https://huggingface.co/litert-community/Gemma3-1B-IT/resolve/main/Gemma3-1B-IT_multi-prefill-seq_q8_ekv1280.task?download=true",
            category = "text",
            sizeBytes = 1050000000L, // 1.05 GB
            source = "Google via LiteRT Community",
            supportsVision = false,
            requirements = ModelRequirements(minRamGB = 3, recommendedRamGB = 4),
            contextWindowSize = 1280,
            modelFormat = "task"
        ),
        LLMModel(
            name = "Gemma-3 1B (INT8, 2k)",
            description = "Higher quality INT8 version of Gemma-3 1B with a 2k context window. Ready to download from HuggingFace (1.07GB)",
            url = "https://huggingface.co/litert-community/Gemma3-1B-IT/resolve/main/Gemma3-1B-IT_multi-prefill-seq_q8_ekv2048.task?download=true",
            category = "text",
            sizeBytes = 1070000000L, // 1.07 GB
            source = "Google via LiteRT Community",
            supportsVision = false,
            requirements = ModelRequirements(minRamGB = 3, recommendedRamGB = 4),
            contextWindowSize = 2048,
            modelFormat = "task"
        ),
        LLMModel(
            name = "Gemma-3 1B (INT8, 4k)",
            description = "Higher quality INT8 version of Gemma-3 1B with a large 4k context window. Ready to download from HuggingFace (1.05GB)",
            url = "https://huggingface.co/litert-community/Gemma3-1B-IT/resolve/main/Gemma3-1B-IT_multi-prefill-seq_q8_ekv4096.task?download=true",
            category = "text",
            sizeBytes = 1050000000L, // 1.05 GB
            source = "Google via LiteRT Community",
            supportsVision = false,
            requirements = ModelRequirements(minRamGB = 4, recommendedRamGB = 5),
            contextWindowSize = 4096,
            modelFormat = "task"
        ),

        // LiteRT Community Models (pre-converted .task files)
        LLMModel(
            name = "Llama-3.2 1B (INT8, 1.2k)",
            description = "Meta's Llama 3.2 1B model optimized for on-device inference with INT8 quantization. Ready to download from HuggingFace (1.29GB)",
            url = "https://huggingface.co/litert-community/Llama-3.2-1B-Instruct/resolve/main/Llama-3.2-1B-Instruct_multi-prefill-seq_q8_ekv1280.task?download=true",
            category = "text",
            sizeBytes = 1290000000L, // 1.29 GB
            source = "Meta via LiteRT Community",
            supportsVision = false,
            requirements = ModelRequirements(minRamGB = 3, recommendedRamGB = 4),
            contextWindowSize = 1280,
            modelFormat = "task"
        ),
        
        LLMModel(
            name = "Llama-3.2 3B (INT8, 1.2k)",
            description = "Meta's Llama 3.2 3B model optimized for on-device inference with INT8 quantization. Ready to download from HuggingFace (3.3GB)",
            url = "https://huggingface.co/litert-community/Llama-3.2-3B-Instruct/resolve/main/Llama-3.2-3B-Instruct_multi-prefill-seq_q8_ekv1280.task?download=true",
            category = "text", 
            sizeBytes = 3300000000L, // 3.3 GB
            source = "Meta via LiteRT Community",
            supportsVision = false,
            requirements = ModelRequirements(minRamGB = 4, recommendedRamGB = 6),
            contextWindowSize = 1280,
            modelFormat = "task"
        ),

        LLMModel(
            name = "Phi-4 Mini (INT8, 1.2k)",
            description = "Microsoft's Phi-4 Mini model optimized for on-device inference with INT8 quantization. Ready to download from HuggingFace (3.94GB)",
            url = "https://huggingface.co/litert-community/Phi-4-mini-instruct/resolve/main/Phi-4-mini-instruct_multi-prefill-seq_q8_ekv1280.task?download=true",
            category = "text",
            sizeBytes = 3940000000L, // 3.94 GB
            source = "Microsoft via LiteRT Community",
            supportsVision = false,
            requirements = ModelRequirements(minRamGB = 5, recommendedRamGB = 7),
            contextWindowSize = 1280,
            modelFormat = "task"
        ),

        // Gemma-3n Models (Multimodal - Text + Vision)
        LLMModel(
            name = "Gemma-3n E2B (Vision+Text)",
            description = "Google Gemma-3n E2B with multimodal capabilities (text and vision). Effective 2B parameters with selective parameter activation. Supports 4k context window and multimodal input including text and images. Ready to download from HuggingFace (2.8GB)",
            url = "https://huggingface.co/google/gemma-3n-E2B-it-litert-preview/resolve/main/gemma-3n-E2B-it-int4.task?download=true",
            category = "multimodal",
            sizeBytes = 2800000000L, // 2.8 GB (estimated)
            source = "Google (LiteRT Preview)",
            supportsVision = true,
            requirements = ModelRequirements(minRamGB = 4, recommendedRamGB = 6),
            contextWindowSize = 4096,
            modelFormat = "task"
        ),
        LLMModel(
            name = "Gemma-3n E4B (Vision+Text)",
            description = "Google Gemma-3n E4B with multimodal capabilities (text and vision). Effective 4B parameters with selective parameter activation. Supports 4k context window and multimodal input including text and images. Ready to download from HuggingFace (4.2GB)",
            url = "https://huggingface.co/google/gemma-3n-E4B-it-litert-preview/resolve/main/gemma-3n-E4B-it-int4.task?download=true",
            category = "multimodal",
            sizeBytes = 4200000000L, // 4.2 GB (estimated)
            source = "Google (LiteRT Preview)",
            supportsVision = true,
            requirements = ModelRequirements(minRamGB = 6, recommendedRamGB = 8),
            contextWindowSize = 4096,
            modelFormat = "task"
        )
    )

    /**
     * Current Status and Next Steps
     */
    fun getStatusMessage(): String {
        return """
        🎯 ON-DEVICE AI MODELS FOR ANDROID
        
        📱 READY FOR DIRECT DOWNLOAD (MediaPipe .task format):
        
        ✅ TEXT MODELS:
        
        🔹 GEMMA-3 SERIES (Google):
        • Gemma-3 1B (INT4, 2k context) - 555MB
        • Gemma-3 1B (INT8, 1.2k context) - 1.05GB
        • Gemma-3 1B (INT8, 2k context) - 1.07GB
        • Gemma-3 1B (INT8, 4k context) - 1.05GB
        
        🔹 GEMMA-3N SERIES (Google - Multimodal):
        • Gemma-3n E2B (Vision+Text, 4k context) - 2.8GB
        • Gemma-3n E4B (Vision+Text, 4k context) - 4.2GB
        
        🔹 LLAMA-3.2 SERIES (Meta):
        • Llama-3.2 1B (INT8, 1.2k context) - 1.29GB
        • Llama-3.2 3B (INT8, 1.2k context) - 3.3GB
        
        🔹 PHI-4 SERIES (Microsoft):
        • Phi-4 Mini (INT8, 1.2k context) - 3.94GB
        
        🖼️ VISION MODELS:
        Models with vision support allow you to upload images and ask questions about them!
        
        📥 HOW TO DOWNLOAD:
        Your app's download feature now works with direct HuggingFace URLs!
        Just tap the download button next to any model.
        
        📱 STATUS: Your Android app is ready to download and use all these models!
        The MediaPipe integration should work smoothly now.
        """.trimIndent()
    }
}