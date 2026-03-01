package com.llmhub.llmhub.data

import android.content.Context
import androidx.datastore.preferences.core.Preferences
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.datastore.preferences.core.MutablePreferences
import kotlinx.coroutines.flow.first
import org.json.JSONObject

data class ModelConfig(
    val maxTokens: Int,
    val topK: Int,
    val topP: Float,
    val temperature: Float,
    val backend: String?, // "CPU" or "GPU" or null
    val deviceId: String?, // e.g. "dev0" for NPU, "gpu" for GPU
    val disableVision: Boolean,
    val disableAudio: Boolean,
    val nGpuLayers: Int = 999, // layers offloaded to GPU/NPU (GGUF/Nexa SDK only)
    val enableThinking: Boolean = true // whether thinking/reasoning output is enabled
) {
    fun toJson(): JSONObject {
        val obj = JSONObject()
        obj.put("maxTokens", maxTokens)
        obj.put("topK", topK)
        // JSONObject.put has no overload for Float on some Android runtimes — use Double
        obj.put("topP", topP.toDouble())
        obj.put("temperature", temperature.toDouble())
        if (backend != null) obj.put("backend", backend) else obj.put("backend", JSONObject.NULL)
        if (deviceId != null) obj.put("deviceId", deviceId) else obj.put("deviceId", JSONObject.NULL)
        obj.put("disableVision", disableVision)
        obj.put("disableAudio", disableAudio)
        obj.put("nGpuLayers", nGpuLayers)
        obj.put("enableThinking", enableThinking)
        return obj
    }

    companion object {
        fun fromJson(obj: JSONObject): ModelConfig {
            return ModelConfig(
                maxTokens = obj.optInt("maxTokens", 0),
                topK = obj.optInt("topK", 64),
                topP = obj.optDouble("topP", 0.95).toFloat(),
                temperature = obj.optDouble("temperature", 1.0).toFloat(),
                backend = if (obj.has("backend") && !obj.isNull("backend")) obj.optString("backend") else null,
                deviceId = if (obj.has("deviceId") && !obj.isNull("deviceId")) obj.optString("deviceId") else null,
                disableVision = obj.optBoolean("disableVision", false),
                disableAudio = obj.optBoolean("disableAudio", false),
                nGpuLayers = obj.optInt("nGpuLayers", 999),
                enableThinking = obj.optBoolean("enableThinking", true)
            )
        }
    }
}

class ModelPreferences(private val context: Context) {
    companion object {
        private val MODEL_CONFIGS_KEY = stringPreferencesKey("model_configs_json")
    }

    suspend fun getModelConfig(modelName: String): ModelConfig? {
        val data = context.dataStore.data.first()
        val jsonStr = data[MODEL_CONFIGS_KEY] ?: return null
        try {
            val root = JSONObject(jsonStr)
            if (!root.has(modelName)) return null
            val obj = root.getJSONObject(modelName)
            return ModelConfig.fromJson(obj)
        } catch (_: Exception) {
            return null
        }
    }

    suspend fun setModelConfig(modelName: String, config: ModelConfig) {
        context.dataStore.edit { prefs: MutablePreferences ->
            val current = prefs[MODEL_CONFIGS_KEY] ?: "{}"
            try {
                val root = JSONObject(current)
                root.put(modelName, config.toJson())
                prefs[MODEL_CONFIGS_KEY] = root.toString()
            } catch (e: Exception) {
                val root = JSONObject()
                root.put(modelName, config.toJson())
                prefs[MODEL_CONFIGS_KEY] = root.toString()
            }
        }
    }

    suspend fun removeModelConfig(modelName: String) {
        context.dataStore.edit { prefs: MutablePreferences ->
            val current = prefs[MODEL_CONFIGS_KEY] ?: return@edit
            try {
                val root = JSONObject(current)
                if (root.has(modelName)) {
                    root.remove(modelName)
                    prefs[MODEL_CONFIGS_KEY] = root.toString()
                }
            } catch (_: Exception) {
                // if parsing fails do nothing
            }
        }
    }
}
