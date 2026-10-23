import Foundation
import Observation

/// The Design tab: HTML prototypes, slides, images and motion written by a local model.
@MainActor @Observable
final class DesignService {
    private weak var app: AppModel?
    func attach(_ app: AppModel) { self.app = app }
    /// Name of the open design for the window title ("Design · Crumb & Co"), or "".
    var currentName: String { "" }
}
