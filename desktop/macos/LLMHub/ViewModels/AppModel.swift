import Foundation
import Observation

enum AppSection: Hashable {
    case chat, agent, design, studio
    case settings(SettingsSection)
}

enum SettingsSection: String, CaseIterable, Hashable {
    case general, models, engines, api, speech, studio, agent, websearch, storage
}

/// App-wide state and services.
@MainActor @Observable
final class AppModel {
    static let shared = AppModel()

    let hardware = HardwareInfo.detect()
    let catalog = CatalogDocument.load()
    let hub = HuggingFaceClient()
    let modelStore: ModelStore
    let downloads: DownloadManager
    let engine = LlamaEngine()
    let engines: EngineHub
    let chatStore = ChatStore()
    let chat: ChatViewModel
    let models: ModelsViewModel
    let speech = SpeechService()
    let agent = AgentService()
    let design = DesignService()
    let studio = StudioService()
    private(set) var api: LocalApiServer!

    var section: AppSection = .chat

    private init() {
        modelStore = ModelStore()
        engines = EngineHub(llama: engine)
        downloads = DownloadManager(hub: hub, store: modelStore)
        chat = ChatViewModel()
        models = ModelsViewModel()
        chat.attach(self)
        models.attach(self)
        agent.attach(self)
        design.attach(self)
        studio.attach(self)
        api = LocalApiServer(app: self)
        api.restart()
    }

    /// Design and Studio collapse the sidebar to the design's 64pt icon rail so the canvas gets the room.
    var isCompact: Bool { section == .design || section == .studio }

    var isSettings: Bool {
        if case .settings = section { return true }
        return false
    }

    var hardwareLine: String {
        hardware.gpuBytes > 0
            ? L.f("HardwareSummaryShared", hardware.gpuName, "\(hardware.ramGB)")
            : L.f("HardwareSummaryCpu", "\(hardware.ramGB)")
    }
}
