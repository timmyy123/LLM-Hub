import SwiftUI

/// Dev-mode voice loop screen — uses the phone's mic + speaker to exercise
/// the full STT → LLM → TTS path before BLE transport / Whisper / Kokoro
/// land. Requires a model already loaded via the regular Chat flow.
struct MimoBotTestView: View {
    var onNavigateBack: () -> Void

    @StateObject private var pipeline = VoicePipeline(
        stt: SFSpeechRecognizerSTT(),
        tts: SystemTTS(),
        sink: SpeakerSink()
    )
    @ObservedObject private var backend = LLMBackend.shared

    var body: some View {
        ZStack {
            ApolloLiquidBackground()

            VStack(spacing: 16) {
                header

                ScrollView {
                    VStack(alignment: .leading, spacing: 16) {
                        modelCard
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
}
