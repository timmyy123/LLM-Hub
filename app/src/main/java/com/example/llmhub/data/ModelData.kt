package com.example.llmhub.data

object ModelData {
    val models = listOf(
        // Gemma-3 1B Models
        LLMModel(
            name = "Gemma-3 1B (INT4)",
            description = "Google Gemma-3 1B with INT4 quantization. Optimized for mobile devices with fast inference speed. Ready to download from HuggingFace (529MB)",
            url = "https://huggingface.co/AfiOne/gemma3-1b-it-int4.task/resolve/main/gemma3-1b-it-int4.task?download=true",
            category = "text",
            sizeBytes = 529_000_000L, // 529 MB
            source = "Google via HuggingFace",
            supportsVision = false,
            requirements = ModelRequirements(minRamGB = 2, recommendedRamGB = 3),
            modelFormat = "task"
        ),

        LLMModel(
            name = "Gemma-3 1B (INT8)",
            description = "Higher quality INT8 version of Gemma-3 1B. Better accuracy with larger size. Ready to download from HuggingFace (~1GB)",
            url = "https://huggingface.co/litert-community/Gemma3-1B-IT/resolve/main/gemma3-1b-it-int8.task?download=true",
            category = "text", 
            sizeBytes = 1_005_000_000L, // ~1 GB
            source = "Google via HuggingFace",
            supportsVision = false,
            requirements = ModelRequirements(minRamGB = 3, recommendedRamGB = 4),
            modelFormat = "task"
        ),

        // Gemma-3 4B Models from litert-community
        LLMModel(
            name = "Gemma-3 4B (INT4)",
            description = "4B parameter model with INT4 quantization for better quality and reasoning. Requires more RAM but provides superior performance. Ready to download from HuggingFace (~2.3GB)",
            url = "https://huggingface.co/litert-community/Gemma3-4B-IT/resolve/main/gemma3-4b-it-int4-web.task?download=true",
            category = "text",
            sizeBytes = 2_560_000_000L, // 2.56 GB
            source = "Google via HuggingFace",
            supportsVision = false,
            requirements = ModelRequirements(minRamGB = 4, recommendedRamGB = 6),
            modelFormat = "task"
        ),

        LLMModel(
            name = "Gemma-3 4B (INT8)",
            description = "4B parameter model with INT8 quantization for highest quality. Better accuracy and reasoning capabilities. Ready to download from HuggingFace (~4.3GB)",
            url = "https://huggingface.co/litert-community/Gemma3-4B-IT/resolve/main/gemma3-4b-it-int8-web.task?download=true",
            category = "text",
            sizeBytes = 3_900_000_000L, // 3.9 GB
            source = "Google via HuggingFace",
            supportsVision = false,
            requirements = ModelRequirements(minRamGB = 6, recommendedRamGB = 8),
            modelFormat = "task"
        )
    )

    /**
     * Current Status and Next Steps
     */
    fun getStatusMessage(): String {
        return """
        🎯 GEMMA-3 MODELS FOR ANDROID
        
        📱 READY FOR DIRECT DOWNLOAD (MediaPipe .task format):
        
        ✅ TEXT MODELS:
        • Gemma-3 1B INT4 (529MB) - Fastest, mobile-optimized
        • Gemma-3 1B INT8 (1GB) - Better quality 
        • Gemma-3 4B INT4 (2.3GB) - Superior performance
        • Gemma-3 4B INT8 (4.3GB) - Highest quality
        
        📥 HOW TO DOWNLOAD:
        Your app's download feature now works with direct HuggingFace URLs!
        Just tap the download button next to any model.
        
        📱 STATUS: Your Android app is ready to download and use all these models!
        The MediaPipe integration should work smoothly now.
        """.trimIndent()
    }
} 