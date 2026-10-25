import AppKit
import SwiftUI

/// Studio › Voice: text to speech with Supertonic (SpeechService), saved to ~/Pictures/LLM Hub/Voice.
struct VoiceView: View {
    @Environment(AppModel.self) private var app

    var body: some View {
        @Bindable var studio = app.studio
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                Text(L.s("StudioVoiceHelpAll")).foregroundStyle(Theme.textMuted)
                VStack(alignment: .leading, spacing: 6) {
                    FieldLabel(L.s("StudioVoiceText"))
                    TextEditor(text: $studio.voiceText)
                        .font(.system(size: 14))
                        .scrollContentBackground(.hidden)
                        .padding(8)
                        .frame(minHeight: 140, maxHeight: 320)
                        .glassEffect(.regular, in: .rect(cornerRadius: 10))
                        .accessibilityIdentifier("VoiceTextBox")
                }
                HStack(spacing: 10) {
                    VoicePicker(voice: $studio.voice)
                    Button {
                        studio.speak()
                    } label: {
                        HStack(spacing: 6) {
                            if studio.isSpeaking { ProgressView().controlSize(.small) } else { Image(systemName: "waveform") }
                            Text(L.s("StudioSpeak"))
                        }
                        .fontWeight(.semibold)
                    }
                    .buttonStyle(.glassProminent)
                    .disabled(studio.voiceText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || studio.isSpeaking)
                    .accessibilityIdentifier("SpeakButton")
                }
                .controlSize(.large)
                if !studio.voiceStatus.isEmpty {
                    Text(studio.voiceStatus).font(.system(size: 12)).foregroundStyle(Theme.textMuted).textSelection(.enabled)
                        .accessibilityIdentifier("VoiceStatus")
                }
                if let file = studio.lastSpeech {
                    HStack(spacing: 8) {
                        Text(file.path).font(.system(size: 12, design: .monospaced)).foregroundStyle(Theme.textFaint)
                            .lineLimit(1).truncationMode(.middle).textSelection(.enabled)
                            .accessibilityIdentifier("LastSpeechPath")
                        Button {
                            NSWorkspace.shared.activateFileViewerSelecting([file])
                        } label: {
                            Image(systemName: "folder")
                        }
                        .buttonStyle(.plain)
                        .foregroundStyle(Theme.textMuted)
                        .help(L.s("StudioShowInFinder"))
                    }
                }
            }
            .frame(maxWidth: 760, alignment: .leading)
            .padding(28)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }
}
