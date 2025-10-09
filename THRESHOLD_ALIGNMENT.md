# Threshold Alignment - Consistent Across System ✅

## Overview
Both RagService and ChatViewModel now use **identical, aligned thresholds** to ensure consistent filtering behavior throughout the RAG pipeline.

## Why Alignment Matters

Previously, the thresholds were inconsistent:
- **RagService** (candidate filtering): EmbeddingGemma primary = `0.65`
- **ChatViewModel** (injection decision): EmbeddingGemma primary = `0.50`

This created a problem: A chunk with 0.55 similarity could theoretically pass ChatViewModel but would be rejected by RagService first, creating confusion.

## Aligned Thresholds ✅

### EmbeddingGemma Model
Both RagService and ChatViewModel now use:
- **Primary threshold**: `0.65` (high semantic confidence)
- **Fallback threshold**: `0.30` + `0.02` lexical overlap (moderate semantic + minimal lexical)
- **Logic**: Accept if similarity > 0.65 OR (similarity > 0.30 AND overlap > 0.02)

### Gecko Model (Unchanged)
Both RagService and ChatViewModel use:
- **Primary threshold**: `0.60`/`0.80` (conservative)
- **Fallback threshold**: `0.35`/`0.05` lexical overlap
- **Logic**: Accept if similarity > 0.80 AND overlap > 0.05 OR (similarity > 0.95 AND overlap > 0.005)

## Code Locations

### RagService.kt (lines ~225-235)
```kotlin
if (isEmbeddingGemma) {
    primaryThreshold = 0.65f      // High semantic confidence
    fallbackThreshold = 0.30f     // Moderate semantic
    lexicalThreshold = 0.02       // Minimal lexical requirement
} else {
    primaryThreshold = 0.60f      // Conservative for Gecko
    fallbackThreshold = 0.35f     
    lexicalThreshold = 0.15       // Higher lexical requirement
}
```

### ChatViewModel.kt (lines ~1110-1120)
```kotlin
val shouldInject = if (isEmbeddingGemma) {
    // Match RagService: primary=0.65, fallback=0.30, lexical=0.02
    (topSimilarity > 0.65f) ||  // High semantic alone
    (topSimilarity > 0.30f && topOverlap > 0.02)  // Moderate semantic + minimal lexical
} else {
    // Gecko: keep conservative thresholds
    (topSimilarity > 0.80f && topOverlap > 0.05) ||
    (topSimilarity > 0.95f && topOverlap > 0.005)
}
```

## Decision Flow

### EmbeddingGemma Document Retrieval
1. User asks: "what's in my resume"
2. **RagService filters candidates**:
   - Chunk A: similarity=0.70, overlap=0.01 → ✅ PASS (0.70 > 0.65 primary)
   - Chunk B: similarity=0.45, overlap=0.05 → ✅ PASS (0.45 > 0.30 AND 0.05 > 0.02)
   - Chunk C: similarity=0.25, overlap=0.08 → ❌ FAIL (0.25 < 0.30)
3. RagService returns top 3: [Chunk A, Chunk B, ...]
4. **ChatViewModel injection decision**:
   - Top similarity=0.70, overlap=0.01 → ✅ INJECT (0.70 > 0.65)
5. Documents injected into prompt ✅

### Gecko Document Retrieval (Unchanged)
1. User asks: "what's in my resume"
2. **RagService filters candidates**:
   - Chunk A: similarity=0.85, overlap=0.10 → ✅ PASS (0.85 > 0.60 primary)
   - Chunk B: similarity=0.75, overlap=0.02 → ❌ FAIL (below primary, overlap too low)
3. RagService returns top 3: [Chunk A, ...]
4. **ChatViewModel injection decision**:
   - Top similarity=0.85, overlap=0.10 → ✅ INJECT (0.85 > 0.80 AND 0.10 > 0.05)
5. Documents injected into prompt ✅

## Benefits

✅ **Consistent behavior**: No confusion from mismatched thresholds
✅ **Predictable results**: Same logic throughout the pipeline
✅ **Clear debugging**: Logs show matching threshold values
✅ **Maintainable**: Change thresholds in one place, update both files
✅ **Gecko unchanged**: Conservative Gecko thresholds preserved

## Testing

### Expected Behavior with EmbeddingGemma

**Query**: "what's in my resume" with similarity=0.70, overlap=0.01
```
RagService: ✅ Returns candidates (0.70 > 0.65)
ChatViewModel: ✅ Injects into prompt (0.70 > 0.65)
Result: Success!
```

**Query**: "what's in my resume" with similarity=0.40, overlap=0.03
```
RagService: ✅ Returns candidates (0.40 > 0.30 AND 0.03 > 0.02)
ChatViewModel: ✅ Injects into prompt (0.40 > 0.30 AND 0.03 > 0.02)
Result: Success!
```

**Query**: "what's in my resume" with similarity=0.25, overlap=0.01
```
RagService: ❌ Filters out (0.25 < 0.30 AND 0.01 < 0.02)
ChatViewModel: N/A (never receives candidates)
Result: No documents found (as expected)
```

## Maintenance

When updating thresholds in the future:
1. Update **RagService.kt** `filterSimilarityCandidates()` function
2. Update **ChatViewModel.kt** `shouldInject` logic
3. Ensure both use identical threshold values
4. Test with sample queries to verify alignment
5. Update this document with new values
