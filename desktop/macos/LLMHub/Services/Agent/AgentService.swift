import Foundation
import Observation

/// The Agent tab: Claude Code sessions driven against LLM Hub's local server.
@MainActor @Observable
final class AgentService {
    private weak var app: AppModel?
    func attach(_ app: AppModel) { self.app = app }
}
