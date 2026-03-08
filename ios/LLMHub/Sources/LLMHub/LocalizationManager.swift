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

    var displayNameKey: String {
        switch self {
        case .systemDefault: return "system_default_language"
        case .english: return "language_english"
        case .arabic: return "language_arabic"
        case .german: return "language_german"
        case .spanish: return "language_spanish"
        case .persian: return "language_persian"
        case .french: return "language_french"
        case .hebrew: return "language_hebrew"
        case .indonesian: return "language_indonesian"
        case .italian: return "language_italian"
        case .japanese: return "language_japanese"
        case .korean: return "language_korean"
        case .polish: return "language_polish"
        case .portuguese: return "language_portuguese"
        case .russian: return "language_russian"
        case .turkish: return "language_turkish"
        case .ukrainian: return "language_ukrainian"
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

    var displayNameKey: String {
        switch self {
        case .light: return "theme_light"
        case .dark: return "theme_dark"
        case .system: return "theme_system"
        }
    }

    var colorScheme: ColorScheme? {
        switch self {
        case .light: return .light
        case .dark: return .dark
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

    private var activeLocalizationCode: String {
        if selectedLanguage == .systemDefault {
            let preferred = Locale.preferredLanguages.first ?? "en"
            return preferred.components(separatedBy: "-").first ?? "en"
        }
        return selectedLanguage.rawValue
    }

    var localizationBundle: Bundle {
        let code = activeLocalizationCode
        // Try finding lproj directly in module bundle
        if let path = Bundle.module.path(forResource: code, ofType: "lproj"),
           let bundle = Bundle(path: path) {
            return bundle
        }
        // Fallback to searching subdirectories
        if let url = Bundle.module.url(forResource: "Localizable", withExtension: "strings", subdirectory: nil, localization: code),
           let bundleURL = (url.deletingLastPathComponent()).absoluteURL as URL?,
           let bundle = Bundle(url: bundleURL) {
            return bundle
        }
        return Bundle.module
    }

    func localized(_ key: String) -> String {
        let localizedValue = localizationBundle.localizedString(forKey: key, value: nil, table: nil)
        if localizedValue == key {
            // Last resort: search main bundle
            return NSLocalizedString(key, bundle: Bundle.main, comment: "")
        }
        return localizedValue
    }
}
