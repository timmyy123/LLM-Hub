#if DEBUG
import Foundation
import StableAudioMac

extension DevTools {
    /// End-to-end Studio runs on this Mac, results in the log. They run before `--page`/`--capture`, so pair them with
    /// `--capture … --page studio --exit` (DevTools only starts its task when --capture or --selftest is present):
    /// - `--studiotest <workflowId> [--prompt <text>] [--image <path|latest>]` generates through the real view model:
    ///   `STUDIOTEST <id> in <s>s: <status>; newest <path>`
    /// - `--musictest "<prompt>" [--seconds 10]` renders a clip with Magenta RealTime: `MUSICTEST …` with the WAV and timing
    /// - `--cliptest "<prompt>" [--seconds 10,30] [--clipmodel sm-music|medium] [--seed 42]` makes Stable Audio 3 clips,
    ///   one per length: `CLIPTEST <model> <s>s seed <n> in <s>s (load, encode, sample, decode) peak <p> rms <r>: <path>`
    /// - `--musiclive <seconds>` plays the live jam muted, changes a style halfway, records and exports: `MUSICLIVE …`
    /// - `--voicetest "<text>"` speaks through Studio › Voice (SpeechService): `VOICETEST <voice>: <status>; file <path>`
    /// - `--studiomode image|video|music|voice` picks the Studio tab for captures.
    static func studioTest(_ args: [String]) async {
        let studio = AppModel.shared.studio
        if let mode = value("--studiomode", args) { studio.mode = mode }
        if let id = value("--studiotest", args) {
            await runStudio(id, prompt: value("--prompt", args) ?? "a cozy cafe interior in morning light, film photo", image: value("--image", args))
        }
        if let prompt = value("--musictest", args) {
            await runMusic(prompt, seconds: Double(value("--seconds", args) ?? "") ?? 10)
        }
        if let prompt = value("--cliptest", args) {
            let model = StableAudio3.Model(rawValue: value("--clipmodel", args) ?? "") ?? .smallMusic
            for seconds in (value("--seconds", args) ?? "10").split(separator: ",").compactMap({ Double($0) }) {
                await runClip(prompt, seconds: seconds, model: model, seed: value("--seed", args).flatMap { UInt64($0) })
            }
        }
        if let seconds = value("--musiclive", args) {
            await runMusicLive(seconds: Double(seconds) ?? 20)
        }
        if let text = value("--voicetest", args) {
            studio.mode = "voice"
            studio.voiceText = text
            studio.speak()
            while studio.isSpeaking { try? await Task.sleep(for: .milliseconds(200)) }
            Log.info("VOICETEST \(studio.voice): \(studio.voiceStatus); file \(studio.lastSpeech?.path ?? "-")")
        }
    }

    private static func runStudio(_ id: String, prompt: String, image: String?) async {
        let studio = AppModel.shared.studio
        guard let workflow = ComfyWorkflow.all.first(where: { $0.id == id }) else {
            Log.error("STUDIOTEST unknown workflow \(id); one of \(ComfyWorkflow.all.map(\.id).joined(separator: ", "))")
            return
        }
        studio.mode = workflow.kind
        studio.workflow = workflow
        studio.prompt = prompt
        if let image {
            if image == "latest" {
                studio.loadGallery()
                studio.referenceImage = studio.gallery.first { !$0.isVideo && $0.url.pathExtension.lowercased() == "png" }?.url.path ?? ""
            } else {
                studio.referenceImage = image
            }
            Log.info("STUDIOTEST reference \(studio.referenceImage)")
        }
        let clock = Date()
        let run = Task { await studio.generateNow() }
        let reporter = Task {
            while !Task.isCancelled {
                try? await Task.sleep(for: .seconds(20))
                if Task.isCancelled { break }
                let comfy = studio.comfy.status
                Log.info("STUDIOTEST \(id) \(Int(Date().timeIntervalSince(clock)))s: \(comfy.isEmpty ? studio.status : comfy) (\(Int(studio.progress * 100))%)")
            }
        }
        await run.value
        reporter.cancel()
        Log.info("STUDIOTEST \(id) in \(Int(Date().timeIntervalSince(clock).rounded()))s: \(studio.status); newest \(studio.gallery.first?.url.path ?? "-")")
    }

    private static func runMusic(_ prompt: String, seconds: Double) async {
        let music = AppModel.shared.studio.music
        let clock = Date()
        let reporter = Task {
            while !Task.isCancelled {
                try? await Task.sleep(for: .seconds(20))
                if Task.isCancelled { break }
                Log.info("MUSICTEST \(Int(Date().timeIntervalSince(clock)))s: \(music.status)")
            }
        }
        do {
            let result = try await music.render(prompt: prompt, seconds: seconds)
            Log.info(String(format: "MUSICTEST \"%@\" %.1fs of audio in %.1fs (%.2fx real time, %.0fs incl. setup): %@",
                            prompt, result.seconds, result.took, result.seconds / max(0.01, result.took), Date().timeIntervalSince(clock), result.url.path))
        } catch {
            Log.error("MUSICTEST failed", error)
        }
        reporter.cancel()
    }

    private static func runClip(_ prompt: String, seconds: Double, model: StableAudio3.Model, seed: UInt64?) async {
        let clips = AppModel.shared.studio.clips
        let clock = Date()
        let reporter = Task {
            while !Task.isCancelled {
                try? await Task.sleep(for: .seconds(20))
                if Task.isCancelled { break }
                Log.info("CLIPTEST \(Int(Date().timeIntervalSince(clock)))s: \(clips.status)")
            }
        }
        do {
            let r = try await clips.render(prompt: prompt, seconds: seconds, seed: seed, model: model)
            Log.info(String(format: "CLIPTEST %@ %.0fs seed %llu in %.1fs (load %.1f, encode %.2f, sample %.2f, decode %.2f; %.1fx real time) peak %.3f rms %.3f: %@",
                            model.rawValue, r.seconds, r.seed, r.took, r.timings.load, r.timings.encode, r.timings.sample, r.timings.decode,
                            r.seconds / max(0.01, r.took), r.peak, r.rms, r.url.path))
        } catch {
            Log.error("CLIPTEST failed", error)
        }
        reporter.cancel()
    }

    private static func runMusicLive(seconds: Double) async {
        let music = AppModel.shared.studio.music
        music.muted = true
        music.isRecording = true
        let clock = Date()
        music.play()
        while !music.isPlaying && (music.isBusy || Date().timeIntervalSince(clock) < 2) { try? await Task.sleep(for: .milliseconds(200)) }
        guard music.isPlaying else {
            Log.error("MUSICLIVE did not start: \(music.status)")
            return
        }
        Log.info("MUSICLIVE started in \(String(format: "%.1f", Date().timeIntervalSince(clock)))s")
        let start = Date()
        var changed = false
        while Date().timeIntervalSince(start) < seconds {
            try? await Task.sleep(for: .seconds(1))
            if !changed && Date().timeIntervalSince(start) > seconds / 2 {
                changed = true
                music.styles[0].weight = 0.1
                music.styles.append(MusicService.Style(text: "Upbeat jazz trio with brushed drums", weight: 0.9))
                music.temperature = 0.4
                Log.info(String(format: "MUSICLIVE style changed at %.1fs played", music.elapsed))
            }
        }
        music.export()
        let exported = music.status
        music.setRecording(false)
        let recorded = music.status
        music.stop()
        while music.isPlaying { try? await Task.sleep(for: .milliseconds(100)) }
        Log.info(String(format: "MUSICLIVE %.1fs played, speed %.2fx real time; %@; %@", music.elapsed, music.speed, exported, recorded))
        music.muted = false
    }
}
#endif
