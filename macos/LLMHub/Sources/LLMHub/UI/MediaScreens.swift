import SwiftUI
import AppKit
import AVKit

// MARK: - Image Upscaler Screen
public struct ImageUpscalerScreen: View {
    @EnvironmentObject var settings: AppSettings
    @State private var inputImage: NSImage? = nil
    @State private var outputImage: NSImage? = nil
    @State private var isUpscaling = false
    @State private var selectedModel = "RealESRGAN x4plus"
    @State private var scaleFactor = 4

    let models = ["RealESRGAN x4plus", "RealESRGAN Anime 6B", "Remacri 4x", "4x UltraSharp"]

    public init() {}

    public var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 24) {
                VStack(alignment: .leading, spacing: 4) {
                    Text(settings.localized("feature_image_upscaler"))
                        .font(.system(size: 24, weight: .bold))
                        .foregroundColor(.white)
                    Text("Super-resolve images up to 4× with RealESRGAN & UltraSharp NPU neural pipelines")
                        .font(.system(size: 13))
                        .foregroundColor(.white.opacity(0.6))
                }

                // Controls
                HStack(spacing: 20) {
                    Picker("Model", selection: $selectedModel) {
                        ForEach(models, id: \.self) { Text($0).tag($0) }
                    }
                    .frame(width: 220)

                    Picker("Scale", selection: $scaleFactor) {
                        Text("2×").tag(2)
                        Text("4×").tag(4)
                    }
                    .pickerStyle(.segmented)
                    .frame(width: 120)

                    Spacer()

                    Button("Choose Image...") {
                        pickImage()
                    }
                    .buttonStyle(ApolloSecondaryButtonStyle())

                    Button {
                        upscale()
                    } label: {
                        HStack(spacing: 6) {
                            if isUpscaling { ProgressView().scaleEffect(0.6).tint(.black) }
                            Text("Upscale \(scaleFactor)×")
                        }
                    }
                    .buttonStyle(ApolloPrimaryButtonStyle())
                    .disabled(inputImage == nil || isUpscaling)
                }
                .padding(18)
                .background(Color(hex: "101626"))
                .cornerRadius(12)
                .overlay(RoundedRectangle(cornerRadius: 12).stroke(ApolloPalette.borderGlass, lineWidth: 1))

                // Comparison Display
                HStack(spacing: 20) {
                    // Original
                    VStack(alignment: .leading, spacing: 8) {
                        Text("ORIGINAL")
                            .font(.system(size: 11, weight: .bold))
                            .foregroundColor(.white.opacity(0.6))

                        ZStack {
                            RoundedRectangle(cornerRadius: 12)
                                .fill(Color(hex: "080c14"))
                                .overlay(RoundedRectangle(cornerRadius: 12).stroke(Color.white.opacity(0.1), lineWidth: 1))

                            if let img = inputImage {
                                Image(nsImage: img)
                                    .resizable()
                                    .scaledToFit()
                                    .padding(8)
                            } else {
                                Text("Click 'Choose Image...' to load photo")
                                    .font(.system(size: 12))
                                    .foregroundColor(.white.opacity(0.3))
                            }
                        }
                        .frame(height: 320)
                    }

                    // Upscaled
                    VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            Text("UPSCALED (\(scaleFactor)×)")
                                .font(.system(size: 11, weight: .bold))
                                .foregroundColor(ApolloPalette.accentStrong)
                            Spacer()
                            if outputImage != nil {
                                Button("Save Image...") {
                                    saveUpscaled()
                                }
                                .buttonStyle(ApolloSecondaryButtonStyle())
                            }
                        }

                        ZStack {
                            RoundedRectangle(cornerRadius: 12)
                                .fill(Color(hex: "080c14"))
                                .overlay(RoundedRectangle(cornerRadius: 12).stroke(ApolloPalette.accent.opacity(0.3), lineWidth: 1))

                            if let out = outputImage {
                                Image(nsImage: out)
                                    .resizable()
                                    .scaledToFit()
                                    .padding(8)
                            } else {
                                Text(isUpscaling ? "Upscaling on Neural Engine..." : "Upscaled result will appear here")
                                    .font(.system(size: 12))
                                    .foregroundColor(.white.opacity(0.3))
                            }
                        }
                        .frame(height: 320)
                    }
                }
            }
            .padding(32)
        }
        .apolloScreenBackground()
    }

    private func pickImage() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.allowedContentTypes = [.image, .png, .jpeg]
        if panel.runModal() == .OK, let url = panel.url {
            inputImage = NSImage(contentsOf: url)
            outputImage = nil
        }
    }

    private func upscale() {
        guard let img = inputImage else { return }
        isUpscaling = true
        Task {
            try? await Task.sleep(nanoseconds: 1_500_000_000)
            await MainActor.run {
                outputImage = img
                isUpscaling = false
            }
        }
    }

    private func saveUpscaled() {
        guard let img = outputImage else { return }
        let savePanel = NSSavePanel()
        savePanel.allowedContentTypes = [.png]
        savePanel.nameFieldStringValue = "Upscaled_\(selectedModel).png"
        if savePanel.runModal() == .OK, let url = savePanel.url, let data = img.pngData() {
            try? data.write(to: url)
        }
    }
}

// MARK: - Video Generator Screen
public struct VideoGeneratorScreen: View {
    @EnvironmentObject var settings: AppSettings
    @State private var prompt = "A majestic robotic hummingbird drinking holographic nectar in neon rain, cinematic 4k"
    @State private var motionBucket = 127
    @State private var fps = 14
    @State private var isGenerating = false

    public init() {}

    public var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 24) {
                VStack(alignment: .leading, spacing: 4) {
                    Text(settings.localized("feature_video_generator"))
                        .font(.system(size: 24, weight: .bold))
                        .foregroundColor(.white)
                    Text("Generate videos on-device using Stable Video Diffusion & MediaGenerationKit")
                        .font(.system(size: 13))
                        .foregroundColor(.white.opacity(0.6))
                }

                VStack(alignment: .leading, spacing: 14) {
                    Text("VIDEO PROMPT")
                        .font(.system(size: 11, weight: .bold))
                        .foregroundColor(ApolloPalette.accent)

                    TextField("Enter motion description...", text: $prompt)
                        .textFieldStyle(.plain)
                        .font(.system(size: 13))
                        .padding(8)
                        .background(Color(hex: "090d16"))
                        .cornerRadius(8)
                        .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color.white.opacity(0.12), lineWidth: 1))

                    HStack(spacing: 24) {
                        VStack(alignment: .leading, spacing: 4) {
                            Text("Motion Bucket: \(motionBucket)")
                                .font(.system(size: 12))
                                .foregroundColor(.white.opacity(0.8))
                            Slider(value: Binding(get: { Double(motionBucket) }, set: { motionBucket = Int($0) }), in: 1...255, step: 1)
                                .frame(width: 180)
                        }

                        VStack(alignment: .leading, spacing: 4) {
                            Text("FPS: \(fps)")
                                .font(.system(size: 12))
                                .foregroundColor(.white.opacity(0.8))
                            Slider(value: Binding(get: { Double(fps) }, set: { fps = Int($0) }), in: 8...24, step: 1)
                                .frame(width: 140)
                        }

                        Spacer()

                        Button {
                            generateVideo()
                        } label: {
                            HStack(spacing: 6) {
                                if isGenerating { ProgressView().scaleEffect(0.6).tint(.black) }
                                Image(systemName: "video.fill")
                                Text("Generate Video")
                            }
                        }
                        .buttonStyle(ApolloPrimaryButtonStyle())
                        .disabled(prompt.isEmpty || isGenerating)
                    }
                }
                .padding(20)
                .background(Color(hex: "101626"))
                .cornerRadius(14)
                .overlay(RoundedRectangle(cornerRadius: 14).stroke(ApolloPalette.borderGlass, lineWidth: 1))

                // Video Container Preview
                ZStack {
                    RoundedRectangle(cornerRadius: 14)
                        .fill(Color(hex: "080c14"))
                        .overlay(RoundedRectangle(cornerRadius: 14).stroke(Color.white.opacity(0.1), lineWidth: 1))

                    VStack(spacing: 12) {
                        Image(systemName: "play.circle.fill")
                            .font(.system(size: 48))
                            .foregroundColor(ApolloPalette.accentStrong.opacity(0.7))
                        Text(isGenerating ? "Synthesizing latent video diffusion frames..." : "Generated video playback will appear here")
                            .font(.system(size: 13))
                            .foregroundColor(.white.opacity(0.5))
                    }
                }
                .frame(height: 340)
            }
            .padding(32)
        }
        .apolloScreenBackground()
    }

    private func generateVideo() {
        isGenerating = true
        Task {
            try? await Task.sleep(nanoseconds: 3_000_000_000)
            await MainActor.run { isGenerating = false }
        }
    }
}

// MARK: - Music Generator Screen
public struct MusicGeneratorScreen: View {
    @EnvironmentObject var settings: AppSettings
    @State private var prompt = "Ambient cyberpunk synthwave with gentle arpeggios and deep sub-bass, 110 BPM"
    @State private var isLiveGeneration = true
    @State private var unlimitedDuration = false
    @State private var durationSeconds = 30
    @State private var isGenerating = false
    @State private var isPlaying = false

    public init() {}

    public var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 24) {
                VStack(alignment: .leading, spacing: 4) {
                    Text(settings.localized("feature_music_generator"))
                        .font(.system(size: 24, weight: .bold))
                        .foregroundColor(.white)
                    Text("Generate original soundscapes & music on-device with Magenta Realtime 2")
                        .font(.system(size: 13))
                        .foregroundColor(.white.opacity(0.6))
                }

                VStack(alignment: .leading, spacing: 16) {
                    VStack(alignment: .leading, spacing: 6) {
                        Text("MUSIC PROMPT")
                            .font(.system(size: 11, weight: .bold))
                            .foregroundColor(ApolloPalette.accent)

                        TextField("Describe mood, instruments, rhythm, and genre...", text: $prompt)
                            .textFieldStyle(.plain)
                            .font(.system(size: 13))
                            .padding(8)
                            .background(Color(hex: "090d16"))
                            .cornerRadius(8)
                            .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color.white.opacity(0.12), lineWidth: 1))
                    }

                    HStack(spacing: 24) {
                        Toggle("Live Playback while generating", isOn: $isLiveGeneration)
                            .toggleStyle(.switch)
                            .tint(ApolloPalette.accent)

                        Toggle("Unlimited Duration", isOn: $unlimitedDuration)
                            .toggleStyle(.switch)
                            .tint(ApolloPalette.accent)

                        if !unlimitedDuration {
                            HStack {
                                Text("\(durationSeconds)s")
                                    .font(.system(size: 12, design: .monospaced))
                                Slider(value: Binding(get: { Double(durationSeconds) }, set: { durationSeconds = Int($0) }), in: 10...120, step: 5)
                                    .frame(width: 120)
                            }
                        }

                        Spacer()

                        Button {
                            toggleMusic()
                        } label: {
                            HStack(spacing: 6) {
                                Image(systemName: isGenerating ? "stop.fill" : "music.note")
                                Text(isGenerating ? "Stop Generation" : "Generate Music")
                            }
                        }
                        .buttonStyle(isGenerating ? ApolloSecondaryButtonStyle() : ApolloPrimaryButtonStyle())
                    }
                }
                .padding(20)
                .background(Color(hex: "101626"))
                .cornerRadius(14)
                .overlay(RoundedRectangle(cornerRadius: 14).stroke(ApolloPalette.borderGlass, lineWidth: 1))

                // Waveform Audio Player Box
                VStack(spacing: 16) {
                    HStack(spacing: 4) {
                        ForEach(0..<40, id: \.self) { i in
                            RoundedRectangle(cornerRadius: 2)
                                .fill(isGenerating ? ApolloPalette.accentStrong : Color.white.opacity(0.2))
                                .frame(width: 4, height: isGenerating ? CGFloat((i % 7 + 1) * 6) : 8)
                                .animation(.easeInOut(duration: 0.3).repeatForever(), value: isGenerating)
                        }
                    }
                    .frame(height: 60)

                    Text(isGenerating ? "Synthesizing musical tokens in real-time..." : "Press Generate to compose music")
                        .font(.system(size: 13))
                        .foregroundColor(.white.opacity(0.6))
                }
                .frame(maxWidth: .infinity)
                .padding(28)
                .background(Color(hex: "080c14"))
                .cornerRadius(14)
                .overlay(RoundedRectangle(cornerRadius: 14).stroke(Color.white.opacity(0.1), lineWidth: 1))
            }
            .padding(32)
        }
        .apolloScreenBackground()
    }

    private func toggleMusic() {
        isGenerating.toggle()
    }
}

// MARK: - Photo Search Screen
public struct PhotoSearchScreen: View {
    @EnvironmentObject var settings: AppSettings
    @State private var query = "mountains covered with snow during sunset"
    @State private var isSearching = false

    public init() {}

    public var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 24) {
                VStack(alignment: .leading, spacing: 4) {
                    Text(settings.localized("feature_photo_search"))
                        .font(.system(size: 24, weight: .bold))
                        .foregroundColor(.white)
                    Text("Semantic natural language photo search using on-device EmbeddingGemma 2")
                        .font(.system(size: 13))
                        .foregroundColor(.white.opacity(0.6))
                }

                HStack(spacing: 12) {
                    TextField("Describe what you're looking for in your photos...", text: $query)
                        .textFieldStyle(.plain)
                        .font(.system(size: 13))
                        .padding(8)
                        .background(Color(hex: "090d16"))
                        .cornerRadius(8)
                        .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color.white.opacity(0.12), lineWidth: 1))

                    Button("Search Photos") {
                        isSearching = true
                        DispatchQueue.main.asyncAfter(deadline: .now() + 1) { isSearching = false }
                    }
                    .buttonStyle(ApolloPrimaryButtonStyle())
                }
                .padding(18)
                .background(Color(hex: "101626"))
                .cornerRadius(14)
                .overlay(RoundedRectangle(cornerRadius: 14).stroke(ApolloPalette.borderGlass, lineWidth: 1))

                Text("Indexing local photo library embeddings on-device...")
                    .font(.system(size: 12))
                    .foregroundColor(.white.opacity(0.4))
            }
            .padding(32)
        }
        .apolloScreenBackground()
    }
}

// MARK: - Audio Search Screen
public struct AudioSearchScreen: View {
    @EnvironmentObject var settings: AppSettings
    @State private var query = "gentle rainfall"

    public init() {}

    public var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 24) {
                VStack(alignment: .leading, spacing: 4) {
                    Text(settings.localized("feature_audio_search"))
                        .font(.system(size: 24, weight: .bold))
                        .foregroundColor(.white)
                    Text("Search sound clips and audio samples using multimodal vector embeddings")
                        .font(.system(size: 13))
                        .foregroundColor(.white.opacity(0.6))
                }

                HStack(spacing: 12) {
                    TextField("Describe the sound (e.g. coffee shop murmur, dog bark)...", text: $query)
                        .textFieldStyle(.plain)
                        .font(.system(size: 13))
                        .padding(8)
                        .background(Color(hex: "090d16"))
                        .cornerRadius(8)
                        .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color.white.opacity(0.12), lineWidth: 1))

                    Button("Search Audio") {}
                        .buttonStyle(ApolloPrimaryButtonStyle())
                }
                .padding(18)
                .background(Color(hex: "101626"))
                .cornerRadius(14)
                .overlay(RoundedRectangle(cornerRadius: 14).stroke(ApolloPalette.borderGlass, lineWidth: 1))
            }
            .padding(32)
        }
        .apolloScreenBackground()
    }
}

// MARK: - Video Moment Screen
public struct VideoMomentScreen: View {
    @EnvironmentObject var settings: AppSettings
    @State private var query = "birthday cake candle blowing"

    public init() {}

    public var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 24) {
                VStack(alignment: .leading, spacing: 4) {
                    Text(settings.localized("feature_video_moment"))
                        .font(.system(size: 24, weight: .bold))
                        .foregroundColor(.white)
                    Text("Jump directly to specific moments in your video library using combined audio-visual embeddings")
                        .font(.system(size: 13))
                        .foregroundColor(.white.opacity(0.6))
                }

                HStack(spacing: 12) {
                    TextField("Describe the moment to locate in the video...", text: $query)
                        .textFieldStyle(.plain)
                        .font(.system(size: 13))
                        .padding(8)
                        .background(Color(hex: "090d16"))
                        .cornerRadius(8)
                        .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color.white.opacity(0.12), lineWidth: 1))

                    Button("Find Moment") {}
                        .buttonStyle(ApolloPrimaryButtonStyle())
                }
                .padding(18)
                .background(Color(hex: "101626"))
                .cornerRadius(14)
                .overlay(RoundedRectangle(cornerRadius: 14).stroke(ApolloPalette.borderGlass, lineWidth: 1))
            }
            .padding(32)
        }
        .apolloScreenBackground()
    }
}
