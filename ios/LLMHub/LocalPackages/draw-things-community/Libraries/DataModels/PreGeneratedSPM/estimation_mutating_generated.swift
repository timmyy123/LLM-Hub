import Dflat
import FlatBuffers
import Foundation
import SQLite3
import SQLiteDflat

// MARK - SQLiteValue for Enumerations

// MARK - Serializer

extension GenerationEstimation: FlatBuffersEncodable {
  public func to(flatBufferBuilder: inout FlatBufferBuilder) -> Offset {
    let start = zzz_DflatGen_GenerationEstimation.startGenerationEstimation(&flatBufferBuilder)
    zzz_DflatGen_GenerationEstimation.add(textEncoded: self.textEncoded, &flatBufferBuilder)
    zzz_DflatGen_GenerationEstimation.add(imageEncoded: self.imageEncoded, &flatBufferBuilder)
    zzz_DflatGen_GenerationEstimation.add(samplingStep: self.samplingStep, &flatBufferBuilder)
    zzz_DflatGen_GenerationEstimation.add(imageDecoded: self.imageDecoded, &flatBufferBuilder)
    zzz_DflatGen_GenerationEstimation.add(
      secondPassImageEncoded: self.secondPassImageEncoded, &flatBufferBuilder)
    zzz_DflatGen_GenerationEstimation.add(
      secondPassSamplingStep: self.secondPassSamplingStep, &flatBufferBuilder)
    zzz_DflatGen_GenerationEstimation.add(
      secondPassImageDecoded: self.secondPassImageDecoded, &flatBufferBuilder)
    zzz_DflatGen_GenerationEstimation.add(imageUpscaled: self.imageUpscaled, &flatBufferBuilder)
    zzz_DflatGen_GenerationEstimation.add(faceRestored: self.faceRestored, &flatBufferBuilder)
    return zzz_DflatGen_GenerationEstimation.endGenerationEstimation(
      &flatBufferBuilder, start: start)
  }
}

extension Optional where Wrapped == GenerationEstimation {
  func to(flatBufferBuilder: inout FlatBufferBuilder) -> Offset {
    self.map { $0.to(flatBufferBuilder: &flatBufferBuilder) } ?? Offset()
  }
}

// MARK - ChangeRequest
