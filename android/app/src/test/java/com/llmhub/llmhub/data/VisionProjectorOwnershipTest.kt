package com.llmhub.llmhub.data

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class VisionProjectorOwnershipTest {
    private fun model(name: String) = ModelData.models.first { it.name == name }

    @Test fun sharedProjectorFileNameBelongsToEveryFamilyThatShipsIt() {
        // Gemma 4 12B, 26B-A4B and 31B all download mmproj-F16.gguf; the first catalog owner is 12B's.
        assertFalse(model("Gemma-4 26B-A4B (QAT Q4_0, GGUF)").isOtherCatalogModelsProjector("mmproj-F16.gguf"))
        assertFalse(model("Gemma-4 31B (UD-IQ2_XXS, GGUF)").isOtherCatalogModelsProjector("mmproj-BF16.gguf"))
    }

    @Test fun anotherFamilysProjectorIsRejected() {
        assertTrue(model("Gemma-4 26B-A4B (QAT Q4_0, GGUF)").isOtherCatalogModelsProjector("mmproj-Inkling-2-Mini-Q8_0.gguf"))
        assertTrue(model("Inkling-2-Mini (UD-Q4_K_XL, GGUF)").isOtherCatalogModelsProjector("mmproj-F16.gguf"))
        assertFalse(model("Inkling-2-Mini (UD-Q4_K_XL, GGUF)").isOtherCatalogModelsProjector("mmproj-Inkling-2-Mini-Q8_0.gguf"))
    }
}
