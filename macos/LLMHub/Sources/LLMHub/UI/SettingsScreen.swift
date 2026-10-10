import SwiftUI
import AppKit

public struct SettingsScreen: View {
    @EnvironmentObject var settings: AppSettings
    @ObservedObject private var modelManager = ModelManager.shared
    @State private var metalGPULayers: Double = 33
    @State private var cpuThreads: Double = Double(ProcessInfo.processInfo.activeProcessorCount)
    @State private var autoReadoutTTS: Bool = false
    @State private var showResetAlert: Bool = false

    var onShowPremium: () -> Void

    public init(onShowPremium: @escaping () -> Void) {
        self.onShowPremium = onShowPremium
    }

    public var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 28) {
                // Title
                VStack(alignment: .leading, spacing: 4) {
                    Text(settings.localized("settings_title"))
                        .font(.system(size: 24, weight: .bold))
                        .foregroundColor(.white)
                    Text("Customize inference acceleration, language, and storage settings")
                        .font(.system(size: 13))
                        .foregroundColor(.white.opacity(0.6))
                }

                // Section 1: Language
                settingSection(title: "Language & Localization", icon: "globe") {
                    VStack(alignment: .leading, spacing: 12) {
                        Text("App Language")
                            .font(.system(size: 13, weight: .medium))
                            .foregroundColor(.white.opacity(0.8))

                        Picker("", selection: $settings.selectedLanguage) {
                            ForEach(AppLanguage.allCases, id: \.self) { lang in
                                Text(lang.displayName).tag(lang)
                            }
                        }
                        .pickerStyle(.menu)
                        .frame(maxWidth: 240)
                    }
                }

                // Section 2: Hardware Acceleration & Metal
                settingSection(title: "Hardware Acceleration & Compute", icon: "cpu") {
                    VStack(alignment: .leading, spacing: 18) {
                        // Metal GPU layers
                        VStack(alignment: .leading, spacing: 8) {
                            HStack {
                                Text("Metal GPU Offload Layers")
                                    .font(.system(size: 13, weight: .medium))
                                    .foregroundColor(.white.opacity(0.85))
                                Spacer()
                                Text("\(Int(metalGPULayers)) layers")
                                    .font(.system(size: 12, design: .monospaced))
                                    .foregroundColor(ApolloPalette.accentStrong)
                            }

                            Slider(value: $metalGPULayers, in: 0...99, step: 1)
                                .tint(ApolloPalette.accent)

                            Text("Offload transformer layers to Apple Silicon Metal GPU unified memory for ultra-fast tok/s.")
                                .font(.system(size: 11))
                                .foregroundColor(.white.opacity(0.5))
                        }

                        Divider().background(Color.white.opacity(0.06))

                        // CPU Threads
                        VStack(alignment: .leading, spacing: 8) {
                            HStack {
                                Text("CPU Inference Threads")
                                    .font(.system(size: 13, weight: .medium))
                                    .foregroundColor(.white.opacity(0.85))
                                Spacer()
                                Text("\(Int(cpuThreads)) threads")
                                    .font(.system(size: 12, design: .monospaced))
                                    .foregroundColor(ApolloPalette.accentStrong)
                            }

                            Slider(value: $cpuThreads, in: 1...Double(max(2, ProcessInfo.processInfo.activeProcessorCount)), step: 1)
                                .tint(ApolloPalette.accent)
                        }
                    }
                }

                // Section 3: Speech & Audio
                settingSection(title: "Audio & Speech", icon: "waveform") {
                    Toggle("Automatic Text-to-Speech Readout", isOn: $autoReadoutTTS)
                        .toggleStyle(.switch)
                        .tint(ApolloPalette.accent)
                }

                // Section 4: Storage & Cache Management
                settingSection(title: "Storage & Cache", icon: "internaldrive") {
                    VStack(alignment: .leading, spacing: 14) {
                        let downloaded = modelManager.downloadedModels
                        let totalBytes = downloaded.reduce(0) { $0 + (modelManager.fileSize(for: $1) ?? 0) }

                        HStack {
                            Text("Model Storage Usage")
                                .font(.system(size: 13, weight: .medium))
                                .foregroundColor(.white.opacity(0.85))
                            Spacer()
                            Text(ByteCountFormatter.string(fromByteCount: totalBytes, countStyle: .file))
                                .font(.system(size: 13, design: .monospaced))
                                .foregroundColor(ApolloPalette.accentStrong)
                        }

                        HStack(spacing: 12) {
                            Button("Reveal Models Folder") {
                                if let dir = modelManager.modelsDirectoryURL {
                                    NSWorkspace.shared.activateFileViewerSelecting([dir])
                                }
                            }
                            .buttonStyle(ApolloSecondaryButtonStyle())

                            Button("Clear Temporary Caches") {
                                clearCache()
                            }
                            .buttonStyle(ApolloSecondaryButtonStyle())

                            Button("Reset Chat Histories") {
                                showResetAlert = true
                            }
                            .buttonStyle(ApolloSecondaryButtonStyle())
                        }
                    }
                }

                // Section 5: About & Credits
                settingSection(title: "About LLM Hub", icon: "info.circle") {
                    VStack(alignment: .leading, spacing: 12) {
                        HStack {
                            Text("LLM Hub for macOS")
                                .font(.system(size: 14, weight: .semibold))
                                .foregroundColor(.white)
                            Spacer()
                            Text("v4.4.0 (Desktop Native)")
                                .font(.system(size: 12, design: .monospaced))
                                .foregroundColor(.white.opacity(0.5))
                        }

                        Text("100% on-device private AI studio. Powered by llama.cpp, LiteRT-LM, Whisper, and Apple Silicon Metal acceleration.")
                            .font(.system(size: 12))
                            .foregroundColor(.white.opacity(0.6))

                        HStack(spacing: 14) {
                            Button("GitHub Repository") {
                                if let url = URL(string: "https://github.com/timmyy123/LLM-Hub") {
                                    NSWorkspace.shared.open(url)
                                }
                            }
                            .buttonStyle(ApolloSecondaryButtonStyle())

                            Button("Upgrade to Premium") {
                                onShowPremium()
                            }
                            .buttonStyle(ApolloPrimaryButtonStyle())
                        }
                    }
                }
            }
            .padding(32)
        }
        .apolloScreenBackground()
        .alert("Reset Chat Histories?", isPresented: $showResetAlert) {
            Button("Cancel", role: .cancel) {}
            Button("Delete All", role: .destructive) {
                ChatStore.shared.clearAllConversations()
            }
        } message: {
            Text("This will permanently delete all saved chat sessions on this Mac.")
        }
    }

    private func settingSection<Content: View>(title: String, icon: String, @ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(spacing: 8) {
                Image(systemName: icon)
                    .foregroundColor(ApolloPalette.accentStrong)
                Text(title)
                    .font(.system(size: 16, weight: .bold))
                    .foregroundColor(.white)
            }

            content()
                .padding(20)
                .background(
                    RoundedRectangle(cornerRadius: 14)
                        .fill(ApolloPalette.bgSurface.opacity(0.75))
                )
                .overlay(
                    RoundedRectangle(cornerRadius: 14)
                        .stroke(ApolloPalette.borderGlass, lineWidth: 1)
                )
        }
    }

    private func clearCache() {
        let tempDir = FileManager.default.temporaryDirectory
        try? FileManager.default.removeItem(at: tempDir)
    }
}
