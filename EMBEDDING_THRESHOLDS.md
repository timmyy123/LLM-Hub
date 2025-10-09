# Embedding Model Thresholds - FIXED ✅

## Overview

The RAG system now uses **model-specific thresholds** in BOTH the RagService AND ChatViewModel to optimize retrieval accuracy based on which embedding model is being used. This ensures that more powerful models like EmbeddingGemma can leverage their superior semantic understanding.

## The Problem (Now Fixed!)

Previously, the system had TWO filtering stages:
1. **RagService** - Filtered candidates and returned top results
2. **ChatViewModel** - Had ANOTHER filter before injecting into prompt (hardcoded at 0.80+ similarity)

This meant even when RagService correctly returned good EmbeddingGemma results (0.635 similarity), the ChatViewModel would reject them with old Gecko thresholds!

## The Solution

Both filtering stages now use the same model-aware thresholds:

### RagService Thresholds (Candidate Filtering)
- **EmbeddingGemma**: 
  - Primary: `0.50` (trust semantic similarity)
  - Fallback: `0.30` + `0.08` lexical overlap
- **Gecko**:
  - Primary: `0.60` (need higher confidence)
  - Fallback: `0.35` + `0.15` lexical overlap

### ChatViewModel Thresholds (Injection Decision)
- **EmbeddingGemma**:
  - `0.50+ similarity && 0.01+ overlap` (moderate semantic + minimal lexical) OR
  - `0.70+ similarity` (high semantic alone is enough)
- **Gecko**:
  - `0.80+ similarity && 0.05+ overlap` (high semantic + some lexical) OR
  - `0.95+ similarity && 0.005+ overlap` (very high semantic + minimal lexical)

## Changes Made

### 1. EmbeddingService.kt
```kotlin
interface EmbeddingService {
    fun getCurrentModelName(): String?  // NEW: Exposes current model
}

class MediaPipeEmbeddingService {
    private var currentModelName: String? = null  // NEW: Tracks model
    
    private fun getModelNameFromPath(path: String): String {
        return when {
            fileName.contains("embeddinggemma", ignoreCase = true) -> "EmbeddingGemma"
            fileName.contains("Gecko", ignoreCase = true) -> "Gecko"
            else -> "Unknown"
        }
    }
}
```

### 2. RagService.kt
```kotlin
private fun filterSimilarityCandidates(...): List<ContextChunk> {
    val modelName = embeddingService.getCurrentModelName() ?: "Gecko"
    val isEmbeddingGemma = modelName.contains("EmbeddingGemma", ignoreCase = true)
    
    // Set thresholds based on model
    val primaryThreshold: Float
    val fallbackThreshold: Float
    val lexicalThreshold: Double
    
    if (isEmbeddingGemma) {
        primaryThreshold = 0.50f
        fallbackThreshold = 0.30f
        lexicalThreshold = 0.08
    } else {
        primaryThreshold = 0.60f
        fallbackThreshold = 0.35f
        lexicalThreshold = 0.15
    }
    // ... filtering logic
}
```

### 3. RagServiceManager.kt
```kotlin
fun getCurrentEmbeddingModelName(): String? {
    return embeddingService?.getCurrentModelName()  // NEW: Expose to ViewModels
}
```

### 4. ChatViewModel.kt
```kotlin
// Get model name to use appropriate thresholds
val modelName = ragServiceManager.getCurrentEmbeddingModelName() ?: "Gecko"
val isEmbeddingGemma = modelName.contains("EmbeddingGemma", ignoreCase = true)

val shouldInject = if (isEmbeddingGemma) {
    // EmbeddingGemma: trust semantic similarity more
    (topSimilarity > 0.50f && topOverlap > 0.01) ||
    (topSimilarity > 0.70f)
} else {
    // Gecko: need higher thresholds
    (topSimilarity > 0.80f && topOverlap > 0.05) ||
    (topSimilarity > 0.95f && topOverlap > 0.005)
}
```

## Your Example - Now Works! ✅

**Query**: "what's in my resume"

### Previous Behavior ❌
```
RagService: Returns 3 results [0.635, 0.635, 0.619] with EmbeddingGemma thresholds
ChatViewModel: ❌ Rejects all (0.635 < 0.80 hardcoded Gecko threshold)
Result: No documents injected into prompt
```

### New Behavior ✅
```
RagService: Returns 3 results [0.635, 0.635, 0.619] with EmbeddingGemma thresholds
ChatViewModel: ✅ Accepts (0.635 > 0.50 EmbeddingGemma threshold && 0.025 > 0.01 overlap)
Result: Documents successfully injected into prompt!
```

## Why EmbeddingGemma Needs Lower Thresholds

1. **Better semantic understanding**: 300M parameter model vs Gecko's 110M
2. **Richer embeddings**: Captures nuanced meaning beyond keywords
3. **Less reliance on lexical overlap**: Can find relevant content even with different wording
4. **Higher quality similarity scores**: A 0.63 from EmbeddingGemma is more meaningful than 0.63 from Gecko

## Model Detection

The system automatically detects which model is active:
- Checks the embedding model filename during initialization
- Sets `currentModelName` to "EmbeddingGemma" or "Gecko"
- Falls back to "Gecko" (conservative) if unknown
- Both RagService and ChatViewModel query this to set appropriate thresholds

## Logging

Now includes model name in logs:
```
Using EmbeddingGemma thresholds: primary=0.5, fallback=0.3, lexical=0.08
✅ Found 3 relevant document chunks (similarity=0.635, overlap=0.025, model=EmbeddingGemma) - injecting
```

## Testing

Try these queries to see improved semantic matching with EmbeddingGemma:
- "what's in my resume" ✅ Now works!
- "tell me about my experience"
- "what do you know about me"
- "summarize my background"

With Gecko, these would need closer word matches to trigger successfully.
