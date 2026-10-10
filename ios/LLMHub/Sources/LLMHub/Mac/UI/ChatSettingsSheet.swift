import SwiftUI

public struct ChatSettingsSheet: View {
    @Environment(\.dismiss) private var dismiss
    @EnvironmentObject var settings: AppSettings
    @ObservedObject private var backend = LLMBackend.shared

    @Binding var temperature: Double
    @Binding var topP: Double
    @Binding var topK: Int
    @Binding var maxTokens: Int
    @Binding var systemPrompt: String
    @Binding var contextWindow: Int

    public init(
        temperature: Binding<Double>,
        topP: Binding<TopPWrapper>,
        topK: Binding<Int>,
        maxTokens: Binding<Int>,
        systemPrompt: Binding<String>,
        contextWindow: Binding<Int>
    ) {
        self._temperature = temperature
        self._topP = Binding(get: { topP.wrappedValue.value }, set: { topP.wrappedValue.value = $0 })
        self._topK = topK
        self._maxTokens = maxTokens
        self._systemPrompt = systemPrompt
        self._contextWindow = contextWindow
    }

    public init(
        temperature: Binding<Double>,
        topP: Binding<Double>,
        topK: Binding<Int>,
        maxTokens: Binding<Int>,
        systemPrompt: Binding<String>,
        contextWindow: Binding<Int>
    ) {
        self._temperature = temperature
        self._topP = topP
        self._topK = topK
        self._maxTokens = maxTokens
        self._systemPrompt = systemPrompt
        self._contextWindow = contextWindow
    }

    public var body: some View {
        VStack(spacing: 0) {
            // Header
            HStack {
                Text(settings.localized("chat_settings"))
                    .font(.system(size: 16, weight: .bold))
                    .foregroundColor(.white)

                Spacer()

                Button {
                    dismiss()
                } label: {
                    Image(systemName: "xmark.circle.fill")
                        .font(.system(size: 18))
                        .foregroundColor(.white.opacity(0.6))
                }
                .buttonStyle(.plain)
            }
            .padding()
            .background(Color(hex: "131a2a"))

            Divider().background(Color.white.opacity(0.1))

            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    // System Prompt
                    VStack(alignment: .leading, spacing: 8) {
                        Text(settings.localized("system_prompt"))
                            .font(.system(size: 13, weight: .semibold))
                            .foregroundColor(.white.opacity(0.9))

                        TextEditor(text: $systemPrompt)
                            .font(.system(size: 12))
                            .frame(height: 100)
                            .padding(8)
                            .background(Color(hex: "090d16"))
                            .cornerRadius(8)
                            .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color.white.opacity(0.1), lineWidth: 1))
                    }

                    Divider().background(Color.white.opacity(0.08))

                    // Temperature
                    VStack(alignment: .leading, spacing: 6) {
                        HStack {
                            Text(settings.localized("temperature"))
                                .font(.system(size: 13, weight: .medium))
                                .foregroundColor(.white.opacity(0.9))
                            Spacer()
                            Text(String(format: "%.2f", temperature))
                                .font(.system(size: 12, design: .monospaced))
                                .foregroundColor(ApolloPalette.accentStrong)
                        }
                        Slider(value: $temperature, in: 0.0...2.0, step: 0.05)
                            .tint(ApolloPalette.accent)
                    }

                    // Top P
                    VStack(alignment: .leading, spacing: 6) {
                        HStack {
                            Text("Top P")
                                .font(.system(size: 13, weight: .medium))
                                .foregroundColor(.white.opacity(0.9))
                            Spacer()
                            Text(String(format: "%.2f", topP))
                                .font(.system(size: 12, design: .monospaced))
                                .foregroundColor(ApolloPalette.accentStrong)
                        }
                        Slider(value: $topP, in: 0.0...1.0, step: 0.05)
                            .tint(ApolloPalette.accent)
                    }

                    // Top K
                    VStack(alignment: .leading, spacing: 6) {
                        HStack {
                            Text("Top K")
                                .font(.system(size: 13, weight: .medium))
                                .foregroundColor(.white.opacity(0.9))
                            Spacer()
                            Text("\(topK)")
                                .font(.system(size: 12, design: .monospaced))
                                .foregroundColor(ApolloPalette.accentStrong)
                        }
                        Slider(value: Binding(get: { Double(topK) }, set: { topK = Int($0) }), in: 1...100, step: 1)
                            .tint(ApolloPalette.accent)
                    }

                    // Max Tokens
                    VStack(alignment: .leading, spacing: 6) {
                        HStack {
                            Text("Max Tokens")
                                .font(.system(size: 13, weight: .medium))
                                .foregroundColor(.white.opacity(0.9))
                            Spacer()
                            Text("\(maxTokens)")
                                .font(.system(size: 12, design: .monospaced))
                                .foregroundColor(ApolloPalette.accentStrong)
                        }
                        Slider(value: Binding(get: { Double(maxTokens) }, set: { maxTokens = Int($0) }), in: 128...8192, step: 128)
                            .tint(ApolloPalette.accent)
                    }

                    // Context Window
                    VStack(alignment: .leading, spacing: 6) {
                        HStack {
                            Text("Context Window Size")
                                .font(.system(size: 13, weight: .medium))
                                .foregroundColor(.white.opacity(0.9))
                            Spacer()
                            Text("\(contextWindow) tokens")
                                .font(.system(size: 12, design: .monospaced))
                                .foregroundColor(ApolloPalette.accentStrong)
                        }
                        Slider(value: Binding(get: { Double(contextWindow) }, set: { contextWindow = Int($0) }), in: 1024...32768, step: 1024)
                            .tint(ApolloPalette.accent)
                    }
                }
                .padding()
            }

            // Footer
            HStack {
                Button("Reset Defaults") {
                    temperature = 0.7
                    topP = 0.9
                    topK = 40
                    maxTokens = 2048
                    systemPrompt = "You are a helpful, respectful, and honest assistant."
                    contextWindow = 4096
                }
                .buttonStyle(ApolloSecondaryButtonStyle())

                Spacer()

                Button("Done") {
                    dismiss()
                }
                .buttonStyle(ApolloPrimaryButtonStyle())
            }
            .padding()
            .background(Color(hex: "090d16"))
        }
        .frame(width: 440, height: 560)
        .background(Color(hex: "0b0f19"))
    }
}

public struct TopPWrapper {
    public var value: Double
    public init(_ value: Double) {
        self.value = value
    }
}
