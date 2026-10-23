import Foundation
import Observation

enum AppSection: Hashable {
    case chat, agent, design, studio
    case settings(SettingsSection)
}

enum SettingsSection: String, CaseIterable, Hashable {
    case general, models, api
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
        api = LocalApiServer(app: self)
        api.restart()
    }

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
