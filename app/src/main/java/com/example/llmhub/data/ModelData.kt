package com.example.llmhub.data

object ModelData {
    val models = listOf(
        // 📥 DIRECT DOWNLOAD - Ready-to-use MediaPipe models from HuggingFace
        LLMModel(
            name = "Gemma-3 1B (INT4) 📥 Direct Download",
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
            name = "Gemma-3 1B (INT8) 📥 Direct Download",
            description = "Higher quality INT8 version of Gemma-3 1B. Better accuracy with larger size. Ready to download from HuggingFace (~1GB)",
            url = "https://huggingface.co/litert-community/Gemma3-1B-IT/resolve/main/gemma3-1b-it-int8.task?download=true",
            category = "text", 
            sizeBytes = 1_005_000_000L, // ~1 GB
            source = "Google via HuggingFace",
            supportsVision = false,
            requirements = ModelRequirements(minRamGB = 3, recommendedRamGB = 4),
            modelFormat = "task"
        ),

        LLMModel(
            name = "Gemma-3 4B 📥 Direct Download",
            description = "4B parameter model for better quality and reasoning. Requires more RAM but provides superior performance. Ready to download from HuggingFace (~4GB)",
            url = "https://huggingface.co/litert-community/Gemma3-4B-IT/resolve/main/gemma3-4b-it-int4.task?download=true",
            category = "text",
            sizeBytes = 4_000_000_000L, // ~4 GB
            source = "Google via HuggingFace",
            supportsVision = false,
            requirements = ModelRequirements(minRamGB = 6, recommendedRamGB = 8),
            modelFormat = "task"
        ),

        LLMModel(
            name = "Gemma-3 12B 🔧 Conversion Required",
            description = "12B parameter model for highest quality reasoning. Requires conversion from QAT safetensors to MediaPipe .task format (~12GB after conversion). Best for complex tasks but needs powerful hardware.",
            url = "https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized",
            category = "text",
            sizeBytes = 12_000_000_000L, // ~12 GB
            source = "Google via HuggingFace",
            supportsVision = false,
            requirements = ModelRequirements(minRamGB = 16, recommendedRamGB = 20),
            modelFormat = "safetensors" // Needs conversion to .task
        ),

        LLMModel(
            name = "Gemma-3n E2B Vision 📥 Direct Download",
            description = "Multimodal model with vision capabilities (2B effective parameters). Can analyze images and respond to visual questions. Ready to download from HuggingFace (~2.9GB)",
            url = "https://huggingface.co/google/gemma-3n-E2B-it-litert-preview/resolve/main/gemma-3n-e2b-it-int4.task?download=true",
            category = "vision",
            sizeBytes = 2_900_000_000L, // ~2.9 GB
            source = "Google via HuggingFace",
            supportsVision = true,
            requirements = ModelRequirements(minRamGB = 4, recommendedRamGB = 6),
            modelFormat = "task"
        ),

        LLMModel(
            name = "Gemma-3n E4B Vision 📥 Direct Download",
            description = "Larger multimodal model with enhanced vision capabilities (4B effective parameters). Better image understanding and visual reasoning. Ready to download from HuggingFace (~4.2GB)",
            url = "https://huggingface.co/google/gemma-3n-E4B-it-litert-preview/resolve/main/gemma-3n-e4b-it-int4.task?download=true",
            category = "vision",
            sizeBytes = 4_200_000_000L, // ~4.2 GB
            source = "Google via HuggingFace",
            supportsVision = true,
            requirements = ModelRequirements(minRamGB = 6, recommendedRamGB = 8),
            modelFormat = "task"
        )
    )

    /**
     * Current Status and Next Steps
     */
    fun getStatusMessage(): String {
        return """
        🎯 GEMMA-3 MODELS FOR ANDROID - COMPLETE GUIDE
        
        📱 READY FOR DIRECT DOWNLOAD (MediaPipe .task format):
        
        ✅ TEXT MODELS:
        • Gemma-3 1B INT4 (529MB) - Fastest, mobile-optimized
        • Gemma-3 1B INT8 (1GB) - Better quality 
        • Gemma-3 4B INT4 (4GB) - Best balance of speed/quality
        
        🖼️ VISION MODELS (Text + Images):
        • Gemma-3n E2B (2.9GB) - Can analyze images
        • Gemma-3n E4B (4.2GB) - Enhanced image understanding
        
        🔧 CONVERSION REQUIRED:
        • Gemma-3 12B QAT (12GB) - Highest quality (needs conversion)
        
        📥 HOW TO DOWNLOAD:
        Your app's download feature now works with direct HuggingFace URLs!
        Just tap the download button next to any model with "📥 Direct Download"
        
        🚀 QUICK START OPTIONS:
        
        1. **Use Download Feature in App** (Recommended):
           - Open your Android app
           - Go to Models section  
           - Tap download on any "📥 Direct Download" model
           - Models download directly from HuggingFace
        
        2. **Manual Download (if needed)**:
           python public/download_gemma3_mediapipe.py download gemma-3-1b-int4
        
        3. **Convert 12B Model**:
           python public/convert_gemma3_to_mediapipe.py convert gemma-3-12b
        
        📱 STATUS: Your Android app is ready to download and use all these models!
        The MediaPipe integration should work smoothly now.
        
        🔗 All tools are in your 'public/' folder for advanced usage.
        """.trimIndent()
    }
} 