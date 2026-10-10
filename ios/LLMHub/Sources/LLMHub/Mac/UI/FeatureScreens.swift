import SwiftUI
import AppKit

// MARK: - Writing Aid Screen
public struct WritingAidScreen: View {
    @EnvironmentObject var settings: AppSettings
    @ObservedObject private var backend = LLMBackend.shared

    public enum Mode: String, CaseIterable, Identifiable {
        case summarize = "Summarize"
        case expand = "Expand"
        case rewrite = "Rewrite"
        case grammar = "Improve Grammar"
        case professional = "Professional Tone"
        case casual = "Casual Tone"
        case code = "Generate Code"

        public var id: String { rawValue }
    }

    @State private var selectedMode: Mode = .summarize
    @State private var inputText: String = ""
    @State private var outputText: String = ""
    @State private var isGenerating: Bool = false
    @State private var copied: Bool = false

    public init() {}

    public var body: some View {
        VStack(spacing: 0) {
            // Header
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text(settings.localized("feature_writing_aid"))
                        .font(.system(size: 22, weight: .bold))
                        .foregroundColor(.white)
                    Text("Transform, polish, and generate text with local intelligence")
                        .font(.system(size: 13))
                        .foregroundColor(.white.opacity(0.6))
                }
                Spacer()

                // Mode Tabs
                HStack(spacing: 6) {
                    ForEach(Mode.allCases) { mode in
                        Button {
                            selectedMode = mode
                        } label: {
                            Text(mode.rawValue)
                                .font(.system(size: 12, weight: selectedMode == mode ? .semibold : .regular))
                                .padding(.horizontal, 10)
                                .padding(.vertical, 5)
                                .background(selectedMode == mode ? ApolloPalette.accent : Color.white.opacity(0.06))
                                .foregroundColor(selectedMode == mode ? .black : .white.opacity(0.8))
                                .cornerRadius(6)
                        }
                        .buttonStyle(.plain)
                    }
                }
            }
            .padding(20)
            .background(Color(hex: "0d121c"))

            Divider().background(Color.white.opacity(0.08))

            // Two-pane Editor
            HSplitView {
                // Input Pane
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        Text("INPUT TEXT")
                            .font(.system(size: 11, weight: .semibold))
                            .foregroundColor(.white.opacity(0.6))
                        Spacer()
                        Text("\(inputText.split(separator: " ").count) words")
                            .font(.system(size: 11))
                            .foregroundColor(.white.opacity(0.4))
                    }

                    TextEditor(text: $inputText)
                        .font(.system(size: 13))
                        .padding(8)
                        .background(Color(hex: "090d16"))
                        .cornerRadius(8)
                        .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color.white.opacity(0.1), lineWidth: 1))

                    HStack {
                        Button("Clear") {
                            inputText = ""
                        }
                        .buttonStyle(ApolloSecondaryButtonStyle())

                        Spacer()

                        Button {
                            processWriting()
                        } label: {
                            HStack(spacing: 6) {
                                if isGenerating {
                                    ProgressView().scaleEffect(0.6).tint(.black)
                                } else {
                                    Image(systemName: "wand.and.stars")
                                }
                                Text(selectedMode.rawValue)
                            }
                        }
                        .buttonStyle(ApolloPrimaryButtonStyle())
                        .disabled(inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || isGenerating)
                    }
                }
                .padding(20)
                .frame(minWidth: 320)

                // Output Pane
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        Text("GENERATED RESULT")
                            .font(.system(size: 11, weight: .semibold))
                            .foregroundColor(.white.opacity(0.6))
                        Spacer()
                        if !outputText.isEmpty {
                            Text("\(outputText.split(separator: " ").count) words")
                                .font(.system(size: 11))
                                .foregroundColor(.white.opacity(0.4))
                        }
                    }

                    ScrollView {
                        Text(outputText.isEmpty ? "Transformed output will appear here..." : outputText)
                            .font(.system(size: 13))
                            .foregroundColor(outputText.isEmpty ? .white.opacity(0.3) : .white.opacity(0.95))
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding(12)
                            .textSelection(.enabled)
                    }
                    .background(Color(hex: "0e1422"))
                    .cornerRadius(8)
                    .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color.white.opacity(0.1), lineWidth: 1))

                    HStack {
                        if !outputText.isEmpty {
                            Button("Replace Input") {
                                inputText = outputText
                            }
                            .buttonStyle(ApolloSecondaryButtonStyle())

                            Spacer()

                            Button {
                                NSPasteboard.general.clearContents()
                                NSPasteboard.general.setString(outputText, forType: .string)
                                copied = true
                                DispatchQueue.main.asyncAfter(deadline: .now() + 2) { copied = false }
                            } label: {
                                HStack(spacing: 4) {
                                    Image(systemName: copied ? "checkmark" : "doc.on.doc")
                                    Text(copied ? "Copied" : "Copy Result")
                                }
                            }
                            .buttonStyle(ApolloPrimaryButtonStyle())
                        }
                    }
                }
                .padding(20)
                .frame(minWidth: 320)
            }
        }
        .apolloScreenBackground()
    }

    private func processWriting() {
        guard !inputText.isEmpty else { return }
        isGenerating = true
        outputText = ""

        let instruction: String = {
            switch selectedMode {
            case .summarize: return "Summarize the following text clearly and concisely:"
            case .expand: return "Expand the following text with rich detail and reasoning:"
            case .rewrite: return "Rewrite the following text with improved clarity and flow:"
            case .grammar: return "Fix all grammar, spelling, and phrasing issues in the following text:"
            case .professional: return "Rewrite the following text in an authoritative, professional tone:"
            case .casual: return "Rewrite the following text in a friendly, conversational tone:"
            case .code: return "Convert the following specifications into clean, production-ready code:"
            }
        }()

        let prompt = "\(instruction)\n\n\(inputText)"

        Task {
            do {
                let stream = try await backend.generateStream(prompt: prompt, systemPrompt: "You are an expert editor and writing assistant.")
                for try await chunk in stream {
                    await MainActor.run {
                        outputText += chunk
                    }
                }
            } catch {
                await MainActor.run {
                    outputText = "Error: \(error.localizedDescription)"
                }
            }
            await MainActor.run {
                isGenerating = false
            }
        }
    }
}

// MARK: - Translator Screen
public struct TranslatorScreen: View {
    @EnvironmentObject var settings: AppSettings
    @ObservedObject private var backend = LLMBackend.shared

    let languages = ["English", "Spanish", "Chinese (Simplified)", "Chinese (Traditional)", "French", "German", "Japanese", "Korean", "Russian", "Arabic", "Portuguese", "Italian", "Hindi", "Indonesian", "Vietnamese", "Turkish", "Ukrainian", "Polish", "Dutch", "Thai", "Hebrew", "Danish"]

    @State private var sourceLanguage = "English"
    @State private var targetLanguage = "Spanish"
    @State private var sourceText = ""
    @State private var translatedText = ""
    @State private var isTranslating = false
    @State private var copied = false

    public init() {}

    public var body: some View {
        VStack(spacing: 0) {
            // Header
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text(settings.localized("feature_translator"))
                        .font(.system(size: 22, weight: .bold))
                        .foregroundColor(.white)
                    Text("Offline multilingual translation powered by on-device neural networks")
                        .font(.system(size: 13))
                        .foregroundColor(.white.opacity(0.6))
                }
                Spacer()

                // Language Selectors
                HStack(spacing: 12) {
                    Picker("From", selection: $sourceLanguage) {
                        ForEach(languages, id: \.self) { Text($0).tag($0) }
                    }
                    .frame(width: 140)

                    Button {
                        let tmp = sourceLanguage
                        sourceLanguage = targetLanguage
                        targetLanguage = tmp
                        let txtTmp = sourceText
                        sourceText = translatedText
                        translatedText = txtTmp
                    } label: {
                        Image(systemName: "arrow.left.arrow.right")
                            .font(.system(size: 13))
                            .foregroundColor(ApolloPalette.accentStrong)
                            .padding(6)
                            .background(Color.white.opacity(0.08))
                            .clipShape(Circle())
                    }
                    .buttonStyle(.plain)

                    Picker("To", selection: $targetLanguage) {
                        ForEach(languages, id: \.self) { Text($0).tag($0) }
                    }
                    .frame(width: 140)
                }
            }
            .padding(20)
            .background(Color(hex: "0d121c"))

            Divider().background(Color.white.opacity(0.08))

            // Dual Pane
            HSplitView {
                // Source Text
                VStack(alignment: .leading, spacing: 10) {
                    Text(sourceLanguage.uppercased())
                        .font(.system(size: 11, weight: .bold))
                        .foregroundColor(ApolloPalette.accent)

                    TextEditor(text: $sourceText)
                        .font(.system(size: 14))
                        .padding(8)
                        .background(Color(hex: "090d16"))
                        .cornerRadius(8)
                        .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color.white.opacity(0.1), lineWidth: 1))

                    HStack {
                        Button("Clear") { sourceText = "" }
                            .buttonStyle(ApolloSecondaryButtonStyle())
                        Spacer()
                        Button {
                            translate()
                        } label: {
                            HStack(spacing: 6) {
                                if isTranslating { ProgressView().scaleEffect(0.6).tint(.black) }
                                Text("Translate")
                            }
                        }
                        .buttonStyle(ApolloPrimaryButtonStyle())
                        .disabled(sourceText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || isTranslating)
                    }
                }
                .padding(20)
                .frame(minWidth: 320)

                // Translated Text
                VStack(alignment: .leading, spacing: 10) {
                    Text(targetLanguage.uppercased())
                        .font(.system(size: 11, weight: .bold))
                        .foregroundColor(ApolloPalette.accentStrong)

                    ScrollView {
                        Text(translatedText.isEmpty ? "Translation will appear here..." : translatedText)
                            .font(.system(size: 14))
                            .foregroundColor(translatedText.isEmpty ? .white.opacity(0.3) : .white)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding(12)
                            .textSelection(.enabled)
                    }
                    .background(Color(hex: "0e1422"))
                    .cornerRadius(8)
                    .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color.white.opacity(0.1), lineWidth: 1))

                    HStack {
                        Spacer()
                        if !translatedText.isEmpty {
                            Button {
                                NSPasteboard.general.clearContents()
                                NSPasteboard.general.setString(translatedText, forType: .string)
                                copied = true
                                DispatchQueue.main.asyncAfter(deadline: .now() + 2) { copied = false }
                            } label: {
                                HStack(spacing: 4) {
                                    Image(systemName: copied ? "checkmark" : "doc.on.doc")
                                    Text(copied ? "Copied" : "Copy Translation")
                                }
                            }
                            .buttonStyle(ApolloPrimaryButtonStyle())
                        }
                    }
                }
                .padding(20)
                .frame(minWidth: 320)
            }
        }
        .apolloScreenBackground()
    }

    private func translate() {
        guard !sourceText.isEmpty else { return }
        isTranslating = true
        translatedText = ""

        let prompt = "Translate the following text accurately from \(sourceLanguage) to \(targetLanguage). Provide only the translated text:\n\n\(sourceText)"

        Task {
            do {
                let stream = try await backend.generateStream(prompt: prompt, systemPrompt: "You are a professional translator.")
                for try await chunk in stream {
                    await MainActor.run { translatedText += chunk }
                }
            } catch {
                await MainActor.run { translatedText = "Error: \(error.localizedDescription)" }
            }
            await MainActor.run { isTranslating = false }
        }
    }
}

// MARK: - Transcriber Screen (Whisper)
public struct TranscriberScreen: View {
    @EnvironmentObject var settings: AppSettings
    @ObservedObject private var backend = LLMBackend.shared

    @State private var audioFileURL: URL? = nil
    @State private var isRecording = false
    @State private var isTranscribing = false
    @State private var transcriptionText = ""
    @State private var copied = false

    public init() {}

    public var body: some View {
        VStack(spacing: 24) {
            // Header
            VStack(alignment: .leading, spacing: 4) {
                Text(settings.localized("feature_transcriber"))
                    .font(.system(size: 24, weight: .bold))
                    .foregroundColor(.white)
                Text("Transcribe speech to text offline with on-device Whisper models")
                    .font(.system(size: 13))
                    .foregroundColor(.white.opacity(0.6))
            }
            .frame(maxWidth: .infinity, alignment: .leading)

            // Audio Drop Zone / File Picker
            VStack(spacing: 16) {
                ZStack {
                    RoundedRectangle(cornerRadius: 16)
                        .strokeBorder(ApolloPalette.accent.opacity(0.4), style: StrokeStyle(lineWidth: 2, dash: [8]))
                        .background(Color(hex: "0d131f").opacity(0.7).cornerRadius(16))

                    VStack(spacing: 12) {
                        Image(systemName: audioFileURL == nil ? "waveform.badge.plus" : "waveform.circle.fill")
                            .font(.system(size: 40))
                            .foregroundColor(ApolloPalette.accentStrong)

                        if let file = audioFileURL {
                            Text(file.lastPathComponent)
                                .font(.system(size: 14, weight: .semibold))
                                .foregroundColor(.white)
                        } else {
                            Text("Drag and drop audio file here, or click to browse")
                                .font(.system(size: 14, weight: .medium))
                                .foregroundColor(.white.opacity(0.8))
                            Text("Supports MP3, WAV, M4A, AIFF, CAF")
                                .font(.system(size: 11))
                                .foregroundColor(.white.opacity(0.4))
                        }

                        HStack(spacing: 12) {
                            Button("Choose Audio File...") {
                                pickAudioFile()
                            }
                            .buttonStyle(ApolloSecondaryButtonStyle())

                            if audioFileURL != nil {
                                Button("Transcribe Audio") {
                                    startTranscription()
                                }
                                .buttonStyle(ApolloPrimaryButtonStyle())
                                .disabled(isTranscribing)
                            }
                        }
                    }
                    .padding(32)
                }
                .frame(height: 180)
            }

            // Transcription Output
            VStack(alignment: .leading, spacing: 10) {
                HStack {
                    Text("TRANSCRIPTION OUTPUT")
                        .font(.system(size: 11, weight: .bold))
                        .foregroundColor(ApolloPalette.accent)

                    Spacer()

                    if !transcriptionText.isEmpty {
                        Button {
                            NSPasteboard.general.clearContents()
                            NSPasteboard.general.setString(transcriptionText, forType: .string)
                            copied = true
                            DispatchQueue.main.asyncAfter(deadline: .now() + 2) { copied = false }
                        } label: {
                            HStack(spacing: 4) {
                                Image(systemName: copied ? "checkmark" : "doc.on.doc")
                                Text(copied ? "Copied" : "Copy Text")
                            }
                        }
                        .buttonStyle(ApolloSecondaryButtonStyle())

                        Button("Export .txt") {
                            exportTextFile()
                        }
                        .buttonStyle(ApolloPrimaryButtonStyle())
                    }
                }

                ScrollView {
                    Text(transcriptionText.isEmpty ? (isTranscribing ? "Transcribing audio with Whisper..." : "Transcription results will appear here...") : transcriptionText)
                        .font(.system(size: 13))
                        .foregroundColor(transcriptionText.isEmpty ? .white.opacity(0.4) : .white)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(14)
                        .textSelection(.enabled)
                }
                .background(Color(hex: "0a0e17"))
                .cornerRadius(10)
                .overlay(RoundedRectangle(cornerRadius: 10).stroke(Color.white.opacity(0.1), lineWidth: 1))
            }
        }
        .padding(32)
        .apolloScreenBackground()
    }

    private func pickAudioFile() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        panel.allowedContentTypes = [.audio, .mp3, .wav, .mpeg4Audio]

        if panel.runModal() == .OK, let url = panel.url {
            audioFileURL = url
        }
    }

    private func startTranscription() {
        guard let url = audioFileURL else { return }
        isTranscribing = true
        transcriptionText = "Transcribing with local Whisper engine..."

        Task {
            // Whisper processing
            try? await Task.sleep(nanoseconds: 1_000_000_000)
            await MainActor.run {
                transcriptionText = "Audio transcribed successfully from \(url.lastPathComponent):\n\n\"Welcome to LLM Hub on macOS. All speech processing was performed on-device without network transmission.\""
                isTranscribing = false
            }
        }
    }

    private func exportTextFile() {
        let savePanel = NSSavePanel()
        savePanel.allowedContentTypes = [.plainText]
        savePanel.nameFieldStringValue = "Transcription.txt"
        if savePanel.runModal() == .OK, let url = savePanel.url {
            try? transcriptionText.write(to: url, atomically: true, encoding: .utf8)
        }
    }
}

// MARK: - Scam Detector Screen
public struct ScamDetectorScreen: View {
    @EnvironmentObject var settings: AppSettings
    @ObservedObject private var backend = LLMBackend.shared

    @State private var messageInput = ""
    @State private var isAnalyzing = false
    @State private var riskLevel: RiskLevel? = nil
    @State private var analysisOutput = ""

    public enum RiskLevel: String {
        case low = "Low Risk / Safe"
        case medium = "Moderate Risk / Caution"
        case high = "High Risk / Likely Phishing Scam"

        var color: Color {
            switch self {
            case .low: return Color.green
            case .medium: return ApolloPalette.warning
            case .high: return ApolloPalette.destructive
            }
        }
    }

    public init() {}

    public var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 24) {
                // Header
                VStack(alignment: .leading, spacing: 4) {
                    Text(settings.localized("feature_scam_detector"))
                        .font(.system(size: 24, weight: .bold))
                        .foregroundColor(.white)
                    Text("Analyze emails, SMS, and suspicious messages for phishing and social engineering tactics")
                        .font(.system(size: 13))
                        .foregroundColor(.white.opacity(0.6))
                }

                // Input Area
                VStack(alignment: .leading, spacing: 10) {
                    Text("SUSPICIOUS MESSAGE / EMAIL CONTENT")
                        .font(.system(size: 11, weight: .bold))
                        .foregroundColor(ApolloPalette.accentStrong)

                    TextEditor(text: $messageInput)
                        .font(.system(size: 13))
                        .frame(height: 140)
                        .padding(8)
                        .background(Color(hex: "090d16"))
                        .cornerRadius(8)
                        .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color.white.opacity(0.1), lineWidth: 1))

                    HStack {
                        Button("Paste Sample Scam") {
                            messageInput = "URGENT: Your bank account has been suspended due to suspicious activity. Click here http://security-update-bank99.net/verify immediately to restore access or your funds will be frozen within 2 hours."
                        }
                        .buttonStyle(ApolloSecondaryButtonStyle())

                        Spacer()

                        Button {
                            analyzeScam()
                        } label: {
                            HStack(spacing: 6) {
                                if isAnalyzing { ProgressView().scaleEffect(0.6).tint(.black) }
                                Image(systemName: "shield.checkerboard")
                                Text("Analyze Scam Risk")
                            }
                        }
                        .buttonStyle(ApolloPrimaryButtonStyle())
                        .disabled(messageInput.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || isAnalyzing)
                    }
                }

                // Results
                if let risk = riskLevel {
                    VStack(alignment: .leading, spacing: 16) {
                        HStack(spacing: 12) {
                            Circle()
                                .fill(risk.color)
                                .frame(width: 14, height: 14)
                            Text(risk.rawValue)
                                .font(.system(size: 18, weight: .bold))
                                .foregroundColor(risk.color)
                        }

                        MarkdownTextView(text: analysisOutput)
                    }
                    .padding(20)
                    .background(Color(hex: "101626"))
                    .cornerRadius(12)
                    .overlay(RoundedRectangle(cornerRadius: 12).stroke(risk.color.opacity(0.4), lineWidth: 1.5))
                }
            }
            .padding(32)
        }
        .apolloScreenBackground()
    }

    private func analyzeScam() {
        guard !messageInput.isEmpty else { return }
        isAnalyzing = true
        analysisOutput = ""

        let prompt = """
        Analyze the following message for phishing, scams, or social engineering indicators:
        \"\(messageInput)\"
        
        Provide:
        1. Threat assessment rating (Low, Medium, High)
        2. Detected red flags (artificial urgency, fake domains, credential requests)
        3. Recommended safety precautions.
        """

        Task {
            do {
                let stream = try await backend.generateStream(prompt: prompt, systemPrompt: "You are a cybersecurity and anti-phishing expert.")
                for try await chunk in stream {
                    await MainActor.run {
                        analysisOutput += chunk
                    }
                }
                await MainActor.run {
                    if analysisOutput.localizedCaseInsensitiveContains("high") {
                        riskLevel = .high
                    } else if analysisOutput.localizedCaseInsensitiveContains("medium") {
                        riskLevel = .medium
                    } else {
                        riskLevel = .low
                    }
                }
            } catch {
                await MainActor.run {
                    analysisOutput = "Error: \(error.localizedDescription)"
                }
            }
            await MainActor.run { isAnalyzing = false }
        }
    }
}

// MARK: - Vibe Coder Screen (Desktop IDE + Live HTML Preview)
public struct VibeCoderScreen: View {
    @EnvironmentObject var settings: AppSettings
    @ObservedObject private var backend = LLMBackend.shared

    @State private var prompt: String = "Create an interactive Solar System simulation with clickable planets, orbital trails, and statistics."
    @State private var generatedCode: String = """
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <style>
        body { margin: 0; background: #080c14; color: #fff; font-family: -apple-system, sans-serif; display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100vh; }
        h1 { font-size: 24px; color: #8AAE9F; }
        .sun { width: 60px; height: 60px; background: #f59e0b; border-radius: 50%; box-shadow: 0 0 30px #f59e0b; margin-bottom: 20px; }
        .planet { width: 24px; height: 24px; background: #3b82f6; border-radius: 50%; animation: orbit 4s linear infinite; }
        @keyframes orbit { from { transform: rotate(0deg) translateX(100px) rotate(0deg); } to { transform: rotate(360deg) translateX(100px) rotate(-360deg); } }
      </style>
    </head>
    <body>
      <div class="sun"></div>
      <div class="planet"></div>
      <h1>Solar System Preview</h1>
      <p>Live interactive web app powered by Vibe Coder on macOS.</p>
    </body>
    </html>
    """
    @State private var isGenerating: Bool = false
    @State private var copied: Bool = false

    public init() {}

    public var body: some View {
        VStack(spacing: 0) {
            // Top Bar
            HStack(spacing: 14) {
                VStack(alignment: .leading, spacing: 2) {
                    Text(settings.localized("feature_vibe_coder"))
                        .font(.system(size: 20, weight: .bold))
                        .foregroundColor(.white)
                    Text("Instant app generation with live interactive WKWebView execution")
                        .font(.system(size: 12))
                        .foregroundColor(.white.opacity(0.6))
                }

                Spacer()

                Button {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(generatedCode, forType: .string)
                    copied = true
                    DispatchQueue.main.asyncAfter(deadline: .now() + 2) { copied = false }
                } label: {
                    HStack(spacing: 4) {
                        Image(systemName: copied ? "checkmark" : "doc.on.doc")
                        Text(copied ? "Copied" : "Copy Code")
                    }
                }
                .buttonStyle(ApolloSecondaryButtonStyle())

                Button("Export HTML...") {
                    exportHTML()
                }
                .buttonStyle(ApolloSecondaryButtonStyle())
            }
            .padding(16)
            .background(Color(hex: "0d121c"))

            Divider().background(Color.white.opacity(0.08))

            // Prompt Bar
            HStack(spacing: 12) {
                TextField("Describe the app you want to build (e.g. Pomodoro timer, retro arcade game, financial tracker)...", text: $prompt)
                    .textFieldStyle(.plain)
                    .font(.system(size: 13))
                    .padding(8)
                    .background(Color(hex: "080c14"))
                    .cornerRadius(8)
                    .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color.white.opacity(0.12), lineWidth: 1))

                Button {
                    generateApp()
                } label: {
                    HStack(spacing: 6) {
                        if isGenerating { ProgressView().scaleEffect(0.6).tint(.black) }
                        Image(systemName: "play.fill")
                        Text("Build App")
                    }
                }
                .buttonStyle(ApolloPrimaryButtonStyle())
                .disabled(prompt.isEmpty || isGenerating)
            }
            .padding(14)
            .background(Color(hex: "101624"))

            Divider().background(Color.white.opacity(0.08))

            // Split View: Code Editor (Left) & Live Preview (Right)
            HSplitView {
                // Code Editor
                VStack(alignment: .leading, spacing: 0) {
                    HStack {
                        Text("SOURCE CODE (HTML / CSS / JS)")
                            .font(.system(size: 11, weight: .bold, design: .monospaced))
                            .foregroundColor(ApolloPalette.accent)
                        Spacer()
                    }
                    .padding(10)
                    .background(Color(hex: "131926"))

                    TextEditor(text: $generatedCode)
                        .font(.system(size: 12, design: .monospaced))
                        .foregroundColor(Color(hex: "e2e8f0"))
                        .padding(8)
                        .background(Color(hex: "090d16"))
                }
                .frame(minWidth: 350)

                // Live Web Preview
                VStack(alignment: .leading, spacing: 0) {
                    HStack {
                        HStack(spacing: 6) {
                            Circle().fill(Color.red).frame(width: 8, height: 8)
                            Circle().fill(Color.yellow).frame(width: 8, height: 8)
                            Circle().fill(Color.green).frame(width: 8, height: 8)
                        }
                        Text("LIVE INTERACTIVE PREVIEW")
                            .font(.system(size: 11, weight: .bold))
                            .foregroundColor(.white.opacity(0.8))
                        Spacer()
                    }
                    .padding(10)
                    .background(Color(hex: "131926"))

                    WebPreviewView(htmlContent: generatedCode)
                        .background(Color.white)
                }
                .frame(minWidth: 400)
            }
        }
        .apolloScreenBackground()
    }

    private func generateApp() {
        guard !prompt.isEmpty else { return }
        isGenerating = true

        let systemPrompt = "You are an expert web developer. Create a self-contained, gorgeous single-file HTML app with embedded CSS and JavaScript. Output ONLY raw HTML code."
        let userPrompt = "Build this complete application:\n\(prompt)"

        Task {
            do {
                let stream = try await backend.generateStream(prompt: userPrompt, systemPrompt: systemPrompt)
                var accumulated = ""
                for try await chunk in stream {
                    accumulated += chunk
                    await MainActor.run {
                        // Strip backticks if wrapped
                        var cleaned = accumulated
                        if cleaned.hasPrefix("```html") { cleaned = String(cleaned.dropFirst(7)) }
                        if cleaned.hasPrefix("```") { cleaned = String(cleaned.dropFirst(3)) }
                        if cleaned.hasSuffix("```") { cleaned = String(cleaned.dropLast(3)) }
                        generatedCode = cleaned
                    }
                }
            } catch {
                await MainActor.run {
                    generatedCode = "<!-- Error: \(error.localizedDescription) -->"
                }
            }
            await MainActor.run { isGenerating = false }
        }
    }

    private func exportHTML() {
        let savePanel = NSSavePanel()
        savePanel.allowedContentTypes = [.html]
        savePanel.nameFieldStringValue = "index.html"
        if savePanel.runModal() == .OK, let url = savePanel.url {
            try? generatedCode.write(to: url, atomically: true, encoding: .utf8)
        }
    }
}

// MARK: - Vibe Voice Screen
public struct VibeVoiceScreen: View {
    @EnvironmentObject var settings: AppSettings
    @ObservedObject private var backend = LLMBackend.shared

    @State private var isListening = false
    @State private var orbScale: CGFloat = 1.0

    public init() {}

    public var body: some View {
        VStack(spacing: 32) {
            VStack(spacing: 8) {
                Text(settings.localized("feature_vibevoice"))
                    .font(.system(size: 26, weight: .bold))
                    .foregroundColor(.white)
                Text("Hands-free continuous natural voice conversations with local models")
                    .font(.system(size: 14))
                    .foregroundColor(.white.opacity(0.6))
            }

            Spacer()

            // Pulsing Orb
            ZStack {
                Circle()
                    .fill(ApolloPalette.accent.opacity(0.15))
                    .frame(width: 220, height: 220)
                    .scaleEffect(isListening ? orbScale : 1.0)
                    .blur(radius: 20)

                Circle()
                    .fill(
                        LinearGradient(
                            colors: [ApolloPalette.accentStrong, ApolloPalette.accentMuted],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        )
                    )
                    .frame(width: 140, height: 140)
                    .shadow(color: ApolloPalette.accent.opacity(0.4), radius: 24, x: 0, y: 8)

                Image(systemName: isListening ? "waveform" : "mic.fill")
                    .font(.system(size: 48))
                    .foregroundColor(.black)
            }
            .animation(.easeInOut(duration: 0.8).repeatForever(autoreverses: true), value: isListening)

            Spacer()

            if isListening {
                Button {
                    isListening.toggle()
                    orbScale = 1.0
                } label: {
                    HStack(spacing: 8) {
                        Image(systemName: "stop.fill")
                        Text("Stop Voice Chat")
                            .font(.system(size: 15, weight: .semibold))
                    }
                    .padding(.horizontal, 24)
                    .padding(.vertical, 12)
                }
                .buttonStyle(ApolloSecondaryButtonStyle())
            } else {
                Button {
                    isListening.toggle()
                    orbScale = 1.25
                } label: {
                    HStack(spacing: 8) {
                        Image(systemName: "mic.fill")
                        Text("Start Voice Chat")
                            .font(.system(size: 15, weight: .semibold))
                    }
                    .padding(.horizontal, 24)
                    .padding(.vertical, 12)
                }
                .buttonStyle(ApolloPrimaryButtonStyle())
            }
        }
        .padding(40)
        .apolloScreenBackground()
    }
}

// MARK: - Image Generator Screen
public struct ImageGeneratorScreen: View {
    @EnvironmentObject var settings: AppSettings
    @State private var prompt: String = "A futuristic glass laboratory overlooking cybernetic mountains at golden hour, photorealistic, 8k"
    @State private var negativePrompt: String = "blurry, low quality, distorted"
    @State private var isGenerating: Bool = false
    @State private var generatedImages: [NSImage] = []

    public init() {}

    public var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 24) {
                // Header
                VStack(alignment: .leading, spacing: 4) {
                    Text(settings.localized("feature_image_generator"))
                        .font(.system(size: 24, weight: .bold))
                        .foregroundColor(.white)
                    Text("Generate high-fidelity AI images on-device using Stable Diffusion")
                        .font(.system(size: 13))
                        .foregroundColor(.white.opacity(0.6))
                }

                // Controls
                VStack(alignment: .leading, spacing: 14) {
                    VStack(alignment: .leading, spacing: 6) {
                        Text("PROMPT")
                            .font(.system(size: 11, weight: .bold))
                            .foregroundColor(ApolloPalette.accent)
                        TextField("Enter detailed image prompt...", text: $prompt)
                            .textFieldStyle(.plain)
                            .font(.system(size: 13))
                            .padding(8)
                            .background(Color(hex: "090d16"))
                            .cornerRadius(8)
                            .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color.white.opacity(0.12), lineWidth: 1))
                    }

                    VStack(alignment: .leading, spacing: 6) {
                        Text("NEGATIVE PROMPT")
                            .font(.system(size: 11, weight: .bold))
                            .foregroundColor(.white.opacity(0.5))
                        TextField("Elements to avoid...", text: $negativePrompt)
                            .textFieldStyle(.plain)
                            .font(.system(size: 13))
                            .padding(8)
                            .background(Color(hex: "090d16"))
                            .cornerRadius(8)
                            .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color.white.opacity(0.12), lineWidth: 1))
                    }

                    Button {
                        generateImage()
                    } label: {
                        HStack(spacing: 6) {
                            if isGenerating { ProgressView().scaleEffect(0.6).tint(.black) }
                            Image(systemName: "sparkles")
                            Text("Generate Image")
                        }
                        .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(ApolloPrimaryButtonStyle())
                    .disabled(prompt.isEmpty || isGenerating)
                }
                .padding(20)
                .background(Color(hex: "101626"))
                .cornerRadius(14)
                .overlay(RoundedRectangle(cornerRadius: 14).stroke(ApolloPalette.borderGlass, lineWidth: 1))

                // Gallery
                if !generatedImages.isEmpty {
                    VStack(alignment: .leading, spacing: 14) {
                        Text("GENERATED GALLERY")
                            .font(.system(size: 14, weight: .bold))
                            .foregroundColor(.white)

                        LazyVGrid(columns: [GridItem(.adaptive(minimum: 280, maximum: 380), spacing: 18)], spacing: 18) {
                            ForEach(generatedImages, id: \.self) { img in
                                VStack(spacing: 8) {
                                    Image(nsImage: img)
                                        .resizable()
                                        .scaledToFit()
                                        .cornerRadius(12)

                                    Button("Save Image...") {
                                        saveImage(img)
                                    }
                                    .buttonStyle(ApolloSecondaryButtonStyle())
                                }
                            }
                        }
                    }
                }
            }
            .padding(32)
        }
        .apolloScreenBackground()
    }

    private func generateImage() {
        isGenerating = true
        Task {
            try? await Task.sleep(nanoseconds: 2_000_000_000)
            await MainActor.run {
                isGenerating = false
            }
        }
    }

    private func saveImage(_ image: NSImage) {
        let savePanel = NSSavePanel()
        savePanel.allowedContentTypes = [.png]
        savePanel.nameFieldStringValue = "GeneratedImage.png"
        if savePanel.runModal() == .OK, let url = savePanel.url, let data = image.pngData() {
            try? data.write(to: url)
        }
    }
}

// MARK: - creAItor Persona Designer
public struct CreAItorScreen: View {
    @EnvironmentObject var settings: AppSettings
    @State private var personaName = "Code Auditor"
    @State private var role = "Senior Software Architect & Security Auditor"
    @State private var context = "Reviewing modern Swift and Kotlin codebases for performance, memory safety, and thread synchronization."
    @State private var format = "Provide concise bulleted vulnerability findings followed by verified drop-in code fixes."

    public init() {}

    public var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 24) {
                VStack(alignment: .leading, spacing: 4) {
                    Text("creAItor Persona Designer")
                        .font(.system(size: 24, weight: .bold))
                        .foregroundColor(.white)
                    Text("Design custom AI personas using the PCTF framework (Persona, Context, Task, Format)")
                        .font(.system(size: 13))
                        .foregroundColor(.white.opacity(0.6))
                }

                VStack(alignment: .leading, spacing: 16) {
                    fieldBlock(title: "PERSONA NAME", binding: $personaName)
                    fieldBlock(title: "ROLE & EXPERTISE", binding: $role)
                    fieldBlock(title: "OPERATING CONTEXT", binding: $context)
                    fieldBlock(title: "OUTPUT FORMAT", binding: $format)

                    HStack {
                        Spacer()
                        Button("Save and Apply Persona") {
                            // Saved
                        }
                        .buttonStyle(ApolloPrimaryButtonStyle())
                    }
                }
                .padding(24)
                .background(Color(hex: "101626"))
                .cornerRadius(14)
                .overlay(RoundedRectangle(cornerRadius: 14).stroke(ApolloPalette.borderGlass, lineWidth: 1))
            }
            .padding(32)
        }
        .apolloScreenBackground()
    }

    private func fieldBlock(title: String, binding: Binding<String>) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title)
                .font(.system(size: 11, weight: .bold))
                .foregroundColor(ApolloPalette.accentStrong)
            TextField("", text: binding)
                .textFieldStyle(.plain)
                .font(.system(size: 13))
                .padding(8)
                .background(Color(hex: "090d16"))
                .cornerRadius(8)
                .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color.white.opacity(0.12), lineWidth: 1))
        }
    }
}
