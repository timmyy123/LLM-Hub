package com.llmhub.llmhub.vibecode

internal object CodexConfig {
    /** TOML literal strings preserve URLs. Android JSONObject.quote escapes '/' as '\/'. */
    fun literal(value: String): String {
        require('\'' !in value && '\n' !in value && '\r' !in value)
        return "'$value'"
    }
    fun shellQuote(value: String) = "'" + value.replace("'", "'\"'\"'") + "'"
    fun arguments(baseUrl: String, contextWindow: Int): String = linkedMapOf(
        "model" to "\"llmhub-local\"", "model_provider" to "\"llmhub_local\"",
        "model_providers.llmhub_local.name" to "\"LLM Hub local\"",
        "model_providers.llmhub_local.base_url" to literal(baseUrl),
        "model_providers.llmhub_local.wire_api" to "\"responses\"",
        "model_providers.llmhub_local.requires_openai_auth" to "false",
        "model_providers.llmhub_local.supports_websockets" to "false",
        "model_providers.llmhub_local.request_max_retries" to "0",
        "model_providers.llmhub_local.stream_max_retries" to "0",
        "model_context_window" to contextWindow.toString(),
        "model_auto_compact_token_limit" to (contextWindow * 0.85).toInt().toString(),
        "features.enable_request_compression" to "false",
        // This release's unified executor subscribes after spawning and can lose early output.
        // The shell executor streams stdout/stderr from process creation.
        "features.unified_exec" to "false",
        "features.apps" to "false", "features.multi_agent" to "false",
        "features.code_mode" to "false", "features.code_mode_only" to "false",
        "analytics.enabled" to "false", "feedback.enabled" to "false"
    ).entries.joinToString(" ") { "-c ${shellQuote("${it.key}=${it.value}")}" }
}
