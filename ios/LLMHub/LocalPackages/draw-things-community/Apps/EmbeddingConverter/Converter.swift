import ArgumentParser
import Diffusion
import Foundation
import ModelOp
import ModelZoo
import NNC

@main
struct Converter: ParsableCommand {
  @Option(
    name: .shortAndLong,
    help: "The T.I. emebdding file that is either the safetensors or the PyTorch checkpoint.")
  var file: String
  @Option(name: .shortAndLong, help: "The name of the T.I. embedding.")
  var name: String
  @Option(name: .shortAndLong, help: "The directory to write the output files to.")
  var outputDirectory: String

  private struct Specification: Codable {
    var name: String
    var file: String
    var version: ModelVersion
    var length: Int
  }

  mutating func run() throws {
    ModelZoo.externalUrls = [URL(fileURLWithPath: outputDirectory)]
    let cleanup = Importer.cleanup(filename: name)
    let fileName = cleanup + "_ti_f16.ckpt"
    let (textEmbeddingLength, modelVersion) = try EmbeddingImporter.import(
      downloadedFile: file, name: name, filename: fileName, keyword: cleanup
    )
    let specification = Specification(
      name: name, file: fileName, version: modelVersion, length: textEmbeddingLength)
    let jsonEncoder = JSONEncoder()
    jsonEncoder.keyEncodingStrategy = .convertToSnakeCase
    jsonEncoder.outputFormatting = .prettyPrinted
    let jsonData = try jsonEncoder.encode(specification)
    print(String(decoding: jsonData, as: UTF8.self))
  }
}
