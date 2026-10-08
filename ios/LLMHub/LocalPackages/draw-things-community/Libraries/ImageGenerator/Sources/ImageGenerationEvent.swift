import Diffusion
import Foundation
import NNC

/// Generation updates for an embedded host. The host owns preview rendering.
public enum ImageGenerationEvent {
  case started(
    id: UUID, name: String, version: ModelVersion, prompt: String,
    signposts: Set<ImageGeneratorSignpost>,
    cancel: () -> Void)
  case progress(
    id: UUID, signpost: ImageGeneratorSignpost, signposts: Set<ImageGeneratorSignpost>,
    preview: Tensor<FloatType>?)
  case segmentStarted(id: UUID)
  case finished(id: UUID)
}
