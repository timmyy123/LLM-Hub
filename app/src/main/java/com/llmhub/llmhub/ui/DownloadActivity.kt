package com.llmhub.llmhub.ui

import android.content.Intent
import android.os.Bundle
import android.app.Activity
import com.llmhub.llmhub.service.ModelDownloadService

class DownloadActivity : Activity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // ...existing code...

        // Example: Start download for a model
        val intent = Intent(this, ModelDownloadService::class.java).apply {
            putExtra("modelName", "MyModel")
            putExtra("modelDescription", "A test model for demo.")
            putExtra("modelUrl", "https://huggingface.co/path/to/model.gguf")
            putExtra("modelSize", 123456789L)
            putExtra("modelCategory", "llm")
            putExtra("modelSource", "huggingface")
            putExtra("supportsVision", false)
            putExtra("supportsGpu", false)
            putExtra("minRamGB", 4)
            putExtra("recommendedRamGB", 8)
            putExtra("hfToken", "YOUR_HF_TOKEN_IF_NEEDED")
        }
        startService(intent)
    }
}
