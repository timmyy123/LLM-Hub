import SwiftUI

private enum TTSChoice: String, CaseIterable, Identifiable {
    case system = "System (AVSpeechSynthesizer)"
    case kokoro = "Kokoro-82M (neural)"
    var id: String { rawValue }
}

private let kokoroDefaultVoice = "af_heart"

/// Dev-mode voice loop screen — phone mic + speaker drive STT → LLM → TTS,
/// switchable between the system synthesizer and Kokoro-82M (one-time
/// ~165 MB download).
struct MimoBotTestView: View {
    var onNavigateBack: () -> Void

    @ObservedObject private var backend = LLMBackend.shared
    @StateObject private var pipeline = VoicePipeline(
        stt: SFSpeechRecognizerSTT(),
        tts: SystemTTS(),
        sink: SpeakerSink()
    )

    @State private var ttsChoice: TTSChoice = .system
    @State private var kokoroReady: Bool = KokoroAssets.isReady(voice: kokoroDefaultVoice)
    @State private var downloadStage: String? = nil
    @State private var downloadDone: Int64 = 0
    @State private var downloadTotal: Int64 = -1

    var body: some View {
        ZStack {
            ApolloLiquidBackground()

            VStack(spacing: 16) {
                header

                ScrollView {
                    VStack(alignment: .leading, spacing: 16) {
                        modelCard
                        voiceCard
                        stateCard
                        if !pipeline.lastTranscript.isEmpty { transcriptCard }
                        if !pipeline.lastResponse.isEmpty   { responseCard }
                    }
                    .padding()
                }

                Spacer()
                actionButton
                    .padding(.horizontal)
                    .padding(.bottom, 24)
            }
        }
        .navigationBarBackButtonHidden(true)
        .onChange(of: ttsChoice) { _ in updateTTS() }
        .onChange(of: kokoroReady) { _ in updateTTS() }
    }

    private func updateTTS() {
        switch ttsChoice {
        case .system:
            pipeline.setTTS(SystemTTS())
        case .kokoro:
            if kokoroReady,
               let model = try? KokoroAssets.modelFile(),
               let voice = try? KokoroAssets.voiceFile(kokoroDefaultVoice) {
                pipeline.setTTS(KokoroTTS(modelURL: model, voicePackURL: voice, voiceId: kokoroDefaultVoice))
            } else {
                pipeline.setTTS(SystemTTS())
            }
        }
    }

    private var header: some View {
        HStack {
            Button(action: onNavigateBack) {
                Image(systemName: "chevron.left")
                    .font(.system(size: 17, weight: .semibold))
                    .padding(8)
            }
            Spacer()
            Text("Mimo Bot (test)").font(.headline)
            Spacer()
            Color.clear.frame(width: 24, height: 24)
        }
        .padding(.horizontal)
        .padding(.top, 8)
    }

    private var modelCard: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Model").font(.caption).foregroundColor(.secondary)
            Text(backend.currentlyLoadedModel ?? "(no model loaded)").font(.headline)
            if backend.currentlyLoadedModel == nil {
                Text("Open Chat first to load a model, then come back.")
                    .font(.caption).foregroundColor(.secondary)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding()
        .background(Color(.secondarySystemBackground))
        .cornerRadius(12)
    }

    private var voiceCard: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Voice").font(.caption).foregroundColor(.secondary)
            ForEach(TTSChoice.allCases) { c in
                HStack {
                    Image(systemName: ttsChoice == c ? "circle.inset.filled" : "circle")
                    Text(c.rawValue)
                    Spacer()
                }
                .contentShape(Rectangle())
                .onTapGesture { ttsChoice = c }
            }

            if ttsChoice == .kokoro && !kokoroReady {
                if let stage = downloadStage {
                    Text(stage).font(.caption)
                    if downloadTotal > 0 {
                        ProgressView(value: Double(downloadDone) / Double(downloadTotal))
                    } else {
                        ProgressView()
                    }
                } else {
                    Button("Download Kokoro (~165 MB)") {
                        Task { await downloadKokoro() }
                    }
                    .buttonStyle(.borderedProminent)
                }
            } else if ttsChoice == .kokoro {
                let g2pName = G2PFactory.best().displayName
                Text("G2P: \(g2pName)")
                    .font(.caption.weight(.semibold))
                if g2pName.hasPrefix("dictionary") {
                    Text("Bundled dictionary covers ~150 words. Run scripts/build_espeak_ios.sh and finish the manual setup in docs/espeak-ng-setup.md for full coverage.")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding()
        .background(Color(.secondarySystemBackground))
        .cornerRadius(12)
    }

    private var stateCard: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("State").font(.caption).foregroundColor(.secondary)
            Text(pipeline.state.rawValue.uppercased()).font(.headline)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding()
        .background(Color(.secondarySystemBackground))
        .cornerRadius(12)
    }

    private var transcriptCard: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("You said").font(.caption).foregroundColor(.secondary)
            Text(pipeline.lastTranscript).font(.body)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding()
        .background(Color(.secondarySystemBackground))
        .cornerRadius(12)
    }

    private var responseCard: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Reply").font(.caption).foregroundColor(.secondary)
            Text(pipeline.lastResponse).font(.body)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding()
        .background(Color(.secondarySystemBackground))
        .cornerRadius(12)
    }

    @ViewBuilder
    private var actionButton: some View {
        if pipeline.state == .idle {
            Button {
                pipeline.startTurn()
            } label: {
                Text("Talk")
                    .font(.headline)
                    .frame(maxWidth: .infinity, minHeight: 52)
            }
            .buttonStyle(.borderedProminent)
            .disabled(backend.currentlyLoadedModel == nil)
        } else {
            Button {
                pipeline.cancel()
            } label: {
                Text("Stop")
                    .font(.headline)
                    .frame(maxWidth: .infinity, minHeight: 52)
            }
            .buttonStyle(.bordered)
        }
    }

    private func downloadKokoro() async {
        do {
            for try await p in KokoroAssets.ensure(voice: kokoroDefaultVoice) {
                await MainActor.run {
                    downloadStage = p.stage
                    downloadDone = p.bytesDone
                    downloadTotal = p.bytesTotal
                }
            }
            await MainActor.run {
                kokoroReady = KokoroAssets.isReady(voice: kokoroDefaultVoice)
                downloadStage = nil
            }
        } catch {
            await MainActor.run { downloadStage = "Download failed: \(error.localizedDescription)" }
        }
    }
}
