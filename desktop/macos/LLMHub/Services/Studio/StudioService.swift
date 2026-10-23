import Foundation
import Observation

/// Studio: image and video (ComfyUI), music (Magenta RealTime on MLX) and voice (Supertonic).
@MainActor @Observable
final class StudioService {
    private weak var app: AppModel?
    func attach(_ app: AppModel) { self.app = app }
}
