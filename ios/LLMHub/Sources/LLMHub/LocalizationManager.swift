import Foundation
import SwiftUI

// MARK: - Supported Languages matching Android locale XMLs
enum AppLanguage: String, CaseIterable, Identifiable, Sendable {
    case systemDefault = "system"
    case english = "en"
    case arabic = "ar"
    case german = "de"
    case spanish = "es"
    case persian = "fa"
    case french = "fr"
    case hebrew = "he"
    case indonesian = "id"
    case italian = "it"
    case japanese = "ja"
    case korean = "ko"
    case polish = "pl"
    case portuguese = "pt"
    case russian = "ru"
    case turkish = "tr"
    case ukrainian = "uk"

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .systemDefault: return "System Default"
        case .english:    return "English"
        case .arabic:     return "العربية"
        case .german:     return "Deutsch"
        case .spanish:    return "Español"
        case .persian:    return "فارسی"
        case .french:     return "Français"
        case .hebrew:     return "עברית"
        case .indonesian: return "Indonesia"
        case .italian:    return "Italiano"
        case .japanese:   return "日本語"
        case .korean:     return "한국어"
        case .polish:     return "Polski"
        case .portuguese: return "Português"
        case .russian:    return "Русский"
        case .turkish:    return "Türkçe"
        case .ukrainian:  return "Українська"
        }
    }

    var isRTL: Bool {
        switch self {
        case .arabic, .persian, .hebrew: return true
        default: return false
        }
    }

    var locale: Locale {
        if self == .systemDefault {
            return Locale.current
        }
        return Locale(identifier: self.rawValue)
    }
}

enum AppTheme: String, CaseIterable, Identifiable, Sendable {
    case light = "light"
    case dark = "dark"
    case system = "system"

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .light:  return "Light"
        case .dark:   return "Dark"
        case .system: return "System Default"
        }
    }

    var colorScheme: ColorScheme? {
        switch self {
        case .light:  return .light
        case .dark:   return .dark
        case .system: return nil
        }
    }
}

// MARK: - Settings Store
@MainActor
final class AppSettings: ObservableObject {
    static let shared = AppSettings()

    @Published var selectedLanguage: AppLanguage {
        didSet { UserDefaults.standard.set(selectedLanguage.rawValue, forKey: "app_language") }
    }
    @Published var theme: AppTheme {
        didSet { UserDefaults.standard.set(theme.rawValue, forKey: "app_theme") }
    }
    @Published var streamingEnabled: Bool {
        didSet { UserDefaults.standard.set(streamingEnabled, forKey: "streaming_enabled") }
    }
    @Published var showResultStatus: Bool {
        didSet { UserDefaults.standard.set(showResultStatus, forKey: "show_result_status") }
    }

    private init() {
        let langRaw = UserDefaults.standard.string(forKey: "app_language") ?? "system"
        selectedLanguage = AppLanguage(rawValue: langRaw) ?? .systemDefault
        let themeRaw = UserDefaults.standard.string(forKey: "app_theme") ?? "system"
        theme = AppTheme(rawValue: themeRaw) ?? .system
        streamingEnabled = UserDefaults.standard.bool(forKey: "streaming_enabled")
        showResultStatus = UserDefaults.standard.object(forKey: "show_result_status") as? Bool ?? true
    }
}
