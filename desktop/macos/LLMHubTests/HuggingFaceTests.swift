import XCTest
@testable import LLMHub

/// How a repo's GGUF files become the variants in the catalog picker and the Hugging Face file table.
final class HuggingFaceTests: XCTestCase {
    private func file(_ path: String, _ size: Int64) -> HfFile { HfFile(path: path, size: size, sha256: nil) }

    /// LiquidAI/LFM2.5-1.2B-Instruct-GGUF ships Q4_0 and QAD-Q4_0: two models, not two parts of one.
    func testSameQuantInTwoFilesStaysTwoVariants() {
        let variants = HuggingFaceClient.groupGguf(repo: "LiquidAI/LFM2.5-1.2B-Instruct-GGUF", files: [
            file("LFM2.5-1.2B-Instruct-Q4_0.gguf", 695_751_488),
            file("LFM2.5-1.2B-Instruct-QAD-Q4_0.gguf", 695_755_488),
            file("LFM2.5-1.2B-Instruct-Q4_K_M.gguf", 730_895_168),
            file("README.md", 3544),
        ])
        XCTAssertEqual(variants.map(\.quant), ["Q4_0", "QAD-Q4_0", "Q4_K_M"])
        XCTAssertEqual(variants.map { $0.files.map(\.fileName) },
                       [["LFM2.5-1.2B-Instruct-Q4_0.gguf"], ["LFM2.5-1.2B-Instruct-QAD-Q4_0.gguf"], ["LFM2.5-1.2B-Instruct-Q4_K_M.gguf"]])
        XCTAssertEqual(Set(variants.map(\.modelId)).count, 3, "each file needs its own model id")
    }

    /// The parts of a split model are one variant, in order, with the vision projector beside them.
    func testSplitPartsAndProjector() {
        let variants = HuggingFaceClient.groupGguf(repo: "unsloth/gemma-4-31b-it-GGUF", files: [
            file("Q8_0/gemma-4-31b-it-Q8_0-00002-of-00002.gguf", 10),
            file("Q8_0/gemma-4-31b-it-Q8_0-00001-of-00002.gguf", 20),
            file("gemma-4-31b-it-Q4_K_M.gguf", 25),
            file("mmproj-F16.gguf", 2),
            file("mmproj-BF16.gguf", 2),
            file("mtp-gemma-4-31b-it.gguf", 1),
        ])
        XCTAssertEqual(variants.map(\.quant), ["Q4_K_M", "Q8_0"])
        XCTAssertEqual(variants[1].files.map(\.fileName), ["gemma-4-31b-it-Q8_0-00001-of-00002.gguf", "gemma-4-31b-it-Q8_0-00002-of-00002.gguf"])
        XCTAssertEqual(variants[1].weightBytes, 30)
        XCTAssertEqual(variants[0].mmproj?.fileName, "mmproj-F16.gguf")
    }
}
