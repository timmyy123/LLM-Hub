import ArgumentParser
import Diffusion
import Foundation
import ModelOp
import ModelZoo
import NNC

extension ModelVersion: ExpressibleByArgument {}

@main
struct Converter: ParsableCommand {
  @Option(
    name: .shortAndLong,
    help: "The LoRA file that is either the safetensors or the PyTorch checkpoint.")
  var file: String
  @Option(name: .shortAndLong, help: "The name of the LoRA.")
  var name: String
  @Option(help: "The model version for this LoRA.")
  var version: ModelVersion?
  @Option(help: "The model network scale factor for this LoRA.")
  var scaleFactor: Double?
  @Option(name: .shortAndLong, help: "The directory to write the output files to.")
  var outputDirectory: String

  private struct Specification: Codable {
    var name: String
    var file: String
    var version: ModelVersion
    var TIEmbedding: Bool
    var textEmbeddingLength: Int
    var isLoHa: Bool
  }

  mutating func run() throws {
    ModelZoo.externalUrls = [URL(fileURLWithPath: outputDirectory)]
    let fileName = Importer.cleanup(filename: name) + "_lora_f16.ckpt"
    let scaleFactor = scaleFactor ?? 1.0
    let (modelVersion, didImportTIEmbedding, textEmbeddingLength, isLoHa) = try LoRAImporter.import(
      downloadedFile: file, name: name, filename: fileName, scaleFactor: scaleFactor,
      forceVersion: version
    ) { _ in
    }
    let specification = Specification(
      name: name, file: fileName, version: modelVersion, TIEmbedding: didImportTIEmbedding,
      textEmbeddingLength: textEmbeddingLength, isLoHa: isLoHa)
    let jsonEncoder = JSONEncoder()
    jsonEncoder.keyEncodingStrategy = .convertToSnakeCase
    jsonEncoder.outputFormatting = .prettyPrinted
    let jsonData = try jsonEncoder.encode(specification)
    print(String(decoding: jsonData, as: UTF8.self))
  }
}
