import Dflat
import FlatBuffers
import Foundation
import SQLite3
import SQLiteDflat

public struct GenerationEstimation: Equatable, FlatBuffersDecodable {
  public var textEncoded: Float32
  public var imageEncoded: Float32
  public var samplingStep: Float32
  public var imageDecoded: Float32
  public var secondPassImageEncoded: Float32
  public var secondPassSamplingStep: Float32
  public var secondPassImageDecoded: Float32
  public var imageUpscaled: Float32
  public var faceRestored: Float32
  public init(
    textEncoded: Float32? = 0.0, imageEncoded: Float32? = 0.0, samplingStep: Float32? = 0.0,
    imageDecoded: Float32? = 0.0, secondPassImageEncoded: Float32? = 0.0,
    secondPassSamplingStep: Float32? = 0.0, secondPassImageDecoded: Float32? = 0.0,
    imageUpscaled: Float32? = 0.0, faceRestored: Float32? = 0.0
  ) {
    self.textEncoded = textEncoded ?? 0.0
    self.imageEncoded = imageEncoded ?? 0.0
    self.samplingStep = samplingStep ?? 0.0
    self.imageDecoded = imageDecoded ?? 0.0
    self.secondPassImageEncoded = secondPassImageEncoded ?? 0.0
    self.secondPassSamplingStep = secondPassSamplingStep ?? 0.0
    self.secondPassImageDecoded = secondPassImageDecoded ?? 0.0
    self.imageUpscaled = imageUpscaled ?? 0.0
    self.faceRestored = faceRestored ?? 0.0
  }
  public init(_ obj: zzz_DflatGen_GenerationEstimation) {
    self.textEncoded = obj.textEncoded
    self.imageEncoded = obj.imageEncoded
    self.samplingStep = obj.samplingStep
    self.imageDecoded = obj.imageDecoded
    self.secondPassImageEncoded = obj.secondPassImageEncoded
    self.secondPassSamplingStep = obj.secondPassSamplingStep
    self.secondPassImageDecoded = obj.secondPassImageDecoded
    self.imageUpscaled = obj.imageUpscaled
    self.faceRestored = obj.faceRestored
  }

  public static func from(byteBuffer bb: ByteBuffer) -> Self {
    Self(zzz_DflatGen_GenerationEstimation.getRootAsGenerationEstimation(bb: bb))
  }

  public static func verify(byteBuffer bb: ByteBuffer) -> Bool {
    do {
      var bb = bb
      var verifier = try Verifier(buffer: &bb)
      try ForwardOffset<zzz_DflatGen_GenerationEstimation>.verify(
        &verifier, at: 0, of: zzz_DflatGen_GenerationEstimation.self)
      return true
    } catch {
      return false
    }
  }

  public static var flatBuffersSchemaVersion: String? {
    return nil
  }
}
