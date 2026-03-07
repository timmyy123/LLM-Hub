package com.example.llmhub.utils

data class CodeBlock(
    val language: String?,
    val content: String,
    val isInline: Boolean = false
)
