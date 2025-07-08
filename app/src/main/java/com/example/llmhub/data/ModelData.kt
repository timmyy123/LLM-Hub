package com.example.llmhub.data

object ModelData {
    val models = listOf(
        // Text Models
        LLMModel(
            name = "Gemma 3 1B",
            description = "Google Gemma-3 1B instruction-tuned (quantised Q4_K_M)",
            url = "https://huggingface.co/MaziyarPanahi/gemma-3-1b-it-GGUF/resolve/main/gemma-3-1b-it.Q4_K_S.gguf?download=true",
            category = "text",
            sizeBytes = 0L,
            source = "MaziyarPanahi",
            supportsVision = false,
            requirements = ModelRequirements(minRamGB = 2, recommendedRamGB = 4)
        ),
        LLMModel(
            name = "Llama 3.2 1B",
            description = "Meta Llama-3.2 1B Instruct (Q4_K_M)",
            url = "https://huggingface.co/unsloth/Llama-3.2-1B/resolve/main/llama-3.2-1b-instruct.Q4_K_M.gguf?download=true",
            category = "text",
            sizeBytes = 0L,
            source = "unsloth",
            supportsVision = false,
            requirements = ModelRequirements(minRamGB = 2, recommendedRamGB = 4)
        ),
        LLMModel(
            name = "Llama 3.2 3B",
            description = "Meta Llama-3.2 3B Instruct (Q4_K_M)",
            url = "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/llama-3.2-3b-instruct.Q4_K_M.gguf?download=true",
            category = "text",
            sizeBytes = 0L,
            source = "bartowski",
            supportsVision = false,
            requirements = ModelRequirements(minRamGB = 4, recommendedRamGB = 6)
        ),
        LLMModel(
            name = "Phi-4 Mini",
            description = "Microsoft Phi-4 Mini Instruct (Q4_K_M).",
            url = "https://huggingface.co/Mungert/Phi-4-mini-instruct.gguf/resolve/main/phi-4-mini-q4_k_m.gguf",
            category = "text",
            sizeBytes = 0L,
            source = "Microsoft",
            supportsVision = false,
            requirements = ModelRequirements(minRamGB = 4, recommendedRamGB = 6)
        ),

        // Vision-capable models
        LLMModel(
            name = "Gemma 3 4B",
            description = "Google Gemma-3 4B instruction-tuned (Q4_0)",
            // Official Google mirror requires auth token; pass via authorization header
            url = "https://huggingface.co/MaziyarPanahi/gemma-3-4b-it-GGUF/resolve/main/gemma-3-4b-it.Q4_K_S.gguf?download=true",
            category = "vision",
            sizeBytes = 0L,
            source = "MaziyarPanahi",
            supportsVision = true,
            requirements = ModelRequirements(minRamGB = 6, recommendedRamGB = 8)
        ),
        LLMModel(
            name = "Gemma 3 12B",
            description = "Google Gemma-3 12B instruction-tuned (Q4_0)",
            url = "https://huggingface.co/MaziyarPanahi/gemma-3-12b-it-GGUF/resolve/main/gemma-3-12b-it.Q4_K_S.gguf?download=true",
            category = "vision",
            sizeBytes = 0L,
            source = "MaziyarPanahi",
            supportsVision = true,
            requirements = ModelRequirements(minRamGB = 10, recommendedRamGB = 14)
        )
    )
} 