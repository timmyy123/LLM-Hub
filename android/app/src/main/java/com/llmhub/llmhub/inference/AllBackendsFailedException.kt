package com.llmhub.llmhub.inference

/**
 * Exception thrown when all attempted inference backends (e.g., GenieX and MediaPipe) fail to load a model.
 */
class AllBackendsFailedException(message: String) : Exception(message)
