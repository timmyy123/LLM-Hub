package com.example.llmhub.data

object ModelData {
    val models = listOf(
        // MediaPipe GPU-Optimized Models (Recommended)
        LLMModel(
            name = "Gemma-3 1B Instruction (MediaPipe)",
            description = "Google Gemma-3 1B instruction-tuned, optimized for MediaPipe with GPU acceleration. Fast and efficient.",
            url = "https://storage.googleapis.com/mediapipe-models/llm_inference/gemma-3-1b-it-int4.task",
            category = "text",
            sizeBytes = 1_100_000_000L, // ~1.1 GB
            source = "Google",
            supportsVision = false,
            requirements = ModelRequirements(minRamGB = 3, recommendedRamGB = 4),
            modelFormat = "task"
        ),
        
        LLMModel(
            name = "Gemma-3N 2B Vision (MediaPipe)", 
            description = "Google Gemma-3N 2B with vision capabilities. Supports text + image input with GPU acceleration.",
            url = "https://storage.googleapis.com/mediapipe-models/llm_inference/gemma-3n-E2B-it-int4.task",
            category = "vision",
            sizeBytes = 3_100_000_000L, // ~3.1 GB
            source = "Google",
            supportsVision = true,
            requirements = ModelRequirements(minRamGB = 6, recommendedRamGB = 8),
            modelFormat = "task"
        ),

        // Keep Your Preferred Models (Updated for MediaPipe)
        LLMModel(
            name = "Phi-4 14B Instruction",
            description = "Microsoft Phi-4 14B instruction-tuned model. High reasoning capability. (Requires conversion to .task format)",
            url = "https://huggingface.co/microsoft/Phi-4/resolve/main/model.safetensors",
            category = "text", 
            sizeBytes = 8_400_000_000L, // ~8.4 GB
            source = "Microsoft",
            supportsVision = false,
            requirements = ModelRequirements(minRamGB = 12, recommendedRamGB = 16),
            modelFormat = "gguf" // Note: Needs conversion to .task for MediaPipe
        ),

        LLMModel(
            name = "Gemma-3 12B Instruction", 
            description = "Google Gemma-3 12B instruction-tuned. High quality responses. (Requires conversion to .task format)",
            url = "https://huggingface.co/google/gemma-3-12b-it/resolve/main/model.safetensors",
            category = "text",
            sizeBytes = 12_600_000_000L, // ~12.6 GB
            source = "Google", 
            supportsVision = false,
            requirements = ModelRequirements(minRamGB = 16, recommendedRamGB = 20),
            modelFormat = "gguf" // Note: Needs conversion to .task for MediaPipe
        ),

        // Additional MediaPipe-Ready Models
        LLMModel(
            name = "Gemma-2 2B Instruction (MediaPipe)",
            description = "Google Gemma-2 2B instruction-tuned, MediaPipe optimized with GPU support. Good balance of size and quality.",
            url = "https://storage.googleapis.com/mediapipe-models/llm_inference/gemma-2-2b-it-int4.task",
            category = "text",
            sizeBytes = 2_200_000_000L, // ~2.2 GB
            source = "Google",
            supportsVision = false,
            requirements = ModelRequirements(minRamGB = 4, recommendedRamGB = 6),
            modelFormat = "task"
        ),

        LLMModel(
            name = "Gemma-3N 4B Vision (MediaPipe)",
            description = "Google Gemma-3N 4B with enhanced vision capabilities. High-quality multimodal responses with GPU acceleration.",
            url = "https://storage.googleapis.com/mediapipe-models/llm_inference/gemma-3n-E4B-it-int4.task", 
            category = "vision",
            sizeBytes = 4_400_000_000L, // ~4.4 GB
            source = "Google",
            supportsVision = true,
            requirements = ModelRequirements(minRamGB = 8, recommendedRamGB = 12),
            modelFormat = "task"
        ),

        // Lightweight Options
        LLMModel(
            name = "TinyLlama 1.1B Chat",
            description = "Compact 1.1B parameter model optimized for mobile devices. Fast inference on older hardware.",
            url = "https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/resolve/main/pytorch_model.bin",
            category = "text",
            sizeBytes = 1_100_000_000L, // ~1.1 GB
            source = "TinyLlama Team",
            supportsVision = false,
            requirements = ModelRequirements(minRamGB = 2, recommendedRamGB = 3),
            modelFormat = "gguf" // Note: Needs conversion to .task for MediaPipe
        )
    )
    
    /**
     * Get information about model conversion for MediaPipe.
     * Users need to convert HuggingFace models to .task format.
     */
    fun getConversionInstructions(): String {
        return """
        MediaPipe Model Conversion Instructions:
        
        1. Install MediaPipe Python package:
           pip install mediapipe
           
        2. Convert HuggingFace model to .task format:
           python -c "
           import mediapipe as mp
           from mediapipe.tasks.python.genai import converter
           
           config = converter.ConversionConfig(
               input_ckpt='path/to/model',
               ckpt_format='safetensors',
               output_dir='output_path',
               backend='gpu'  # Enable GPU acceleration
           )
           converter.convert_checkpoint(config)
           "
           
        3. Push the .task file to your device:
           adb push output_path/model.task /data/local/tmp/llm/
           
        For detailed instructions visit:
        https://ai.google.dev/edge/mediapipe/solutions/genai/llm_inference/android
        """.trimIndent()
    }
} 