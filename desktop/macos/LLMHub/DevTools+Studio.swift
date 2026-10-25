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
    /// - `--searchtest photos|audio|videos --folder <dir> [--model embeddinggemma-2-440m] --queries "a red car|/path/photo.jpg"`
    ///   installs the EmbeddingGemma 2 build if needed, indexes the folder in Studio › Search and logs the best matches for
    ///   each query (text, or a photo path for photo-to-photo): `SEARCHTEST "<query>" in <s>s: name score | …`. A query can
    ///   name its right answer after "=>" (a file name, or for videos "file.mp4@20-30", where the best moment must start
    ///   between those seconds): `SEARCHTEST PASS|FAIL …`, then `SEARCHTEST RESULT 4/4 passed`. With `--playmoment`, the
    ///   best video moment then plays, muted: `PLAYTEST <name> from <m:ss>: player at <s>s`.
    /// - `--studiomode image|video|music|voice|search` picks the Studio tab for captures.
    static func studioTest(_ args: [String]) async {
        let studio = AppModel.shared.studio
        if let mode = value("--studiomode", args) { studio.mode = mode }
        if let mode = value("--searchtest", args), let folder = value("--folder", args) {
            await runSearch(mode, folder: folder, catalogId: value("--model", args) ?? "embeddinggemma-2-740m",
                            queries: (value("--queries", args) ?? "").split(separator: "|").map(String.init))
            if args.contains("--playmoment"), let moment = studio.search.videos.first {
                await studio.search.play(moment, muted: true)
                try? await Task.sleep(for: .seconds(3))
                let player = studio.search.videoPlayer
                Log.info("PLAYTEST \(moment.name) from \(moment.label): player at \(String(format: "%.1f", player.currentTime().seconds))s after 3 s, muted \(player.isMuted)")
            }
        }
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

    private static func runSearch(_ mode: String, folder: String, catalogId: String, queries: [String]) async {
        let app = AppModel.shared
        do {
            guard let model = app.catalog.models.first(where: { $0.id == catalogId }),
                  let source = model.sources.first(where: { $0.format == "litertlm" }) else { throw ComfyService.error("\(catalogId) is not in the catalog") }
            // An installed build is used as it is, without asking the Hub.
            var modelId = app.modelStore.installed.first { $0.catalogId == catalogId && $0.format == "litertlm" }?.id
            if modelId == nil {
                let variants = try await app.hub.liteRtVariants(source.repo)
                Log.info("SEARCHTEST variants: \(variants.map { "\($0.quant) \($0.files[0].fileName) \(Fmt.bytes($0.totalBytes))" }.joined(separator: ", "))")
                let variant = variants[0]
                let clock = Date()
                app.downloads.start(variant, name: model.name, catalogId: model.id, tags: model.tags, contextLength: model.contextLength)
                while app.downloads.find(variant.modelId) != nil { try await Task.sleep(for: .seconds(1)) }
                Log.info("SEARCHTEST downloaded \(variant.modelId) (\(Fmt.bytes(variant.totalBytes))) in \(Int(Date().timeIntervalSince(clock)))s")
                modelId = variant.modelId
            }
            let search = app.studio.search
            app.studio.mode = "search"
            search.mode = mode
            search.selectedModelId = modelId
            search.setFolder(folder)
            var clock = Date()
            await search.index()
            Log.info("SEARCHTEST index \(mode) with \(search.selectedModel?.id ?? "-") on \(app.mediaEmbedder.loaded?.backend ?? "-") (\(app.mediaEmbedder.loaded?.maxInput ?? 0) tokens) in \(String(format: "%.1f", Date().timeIntervalSince(clock)))s: \(search.status) (\(search.indexedText))")
            var passed = 0, checked = 0
            for entry in queries {
                let parts = entry.components(separatedBy: "=>")
                let query = parts[0].trimmingCharacters(in: .whitespaces)
                let expected = parts.count > 1 ? parts[1].trimmingCharacters(in: .whitespaces) : nil
                clock = Date()
                if FileManager.default.fileExists(atPath: query) {
                    await search.searchByPhoto(URL(fileURLWithPath: query))
                } else {
                    search.query = query
                    await search.search()
                }
                let top = switch mode {
                case "photos": search.photos.prefix(5).map { "\($0.name) \($0.scoreText)" }
                case "audio": search.audio.prefix(5).map { "\($0.name) \($0.scoreText) @\($0.moments.map(\.label).joined(separator: "/"))" }
                default: search.videos.map { "\($0.name)@\($0.label) \($0.scoreText)" }
                }
                Log.info("SEARCHTEST \"\((query as NSString).lastPathComponent)\" in \(String(format: "%.2f", Date().timeIntervalSince(clock)))s: \(top.joined(separator: " | "))")
                guard let expected else { continue }
                checked += 1
                let (ok, got): (Bool, String?) = switch mode {
                case "photos": (search.photos.first?.name == expected, search.photos.first?.name)
                case "audio": (search.audio.first?.name == expected, search.audio.first?.name)
                default: videoMatches(search.videos.first, expected)
                }
                if ok { passed += 1 }
                Log.info("SEARCHTEST \(ok ? "PASS" : "FAIL") \"\(query)\": expected \(expected), got \(got ?? "nothing")")
            }
            if checked > 0 { Log.info("SEARCHTEST RESULT \(passed)/\(checked) passed") }
            // The shared engine made for video moments (1024 input tokens) also answers photo, audio and API requests:
            // check it embeds a text like the usual 256-token engine.
            if mode == "videos", let longer = app.mediaEmbedder.loaded, longer.maxInput != MediaEmbedder.maxInputTokens,
               let text = queries.first?.components(separatedBy: "=>").first {
                let shorter = try await MediaEmbedder.open(longer.model)
                if let a = await longer.embedText(text, query: true), let b = await shorter.embedText(text, query: true) {
                    Log.info("SEARCHTEST \(longer.maxInput) vs \(shorter.maxInput) input tokens on \(longer.backend)/\(shorter.backend): cosine \(String(format: "%.4f", MediaIndex.dot(a, b)))")
                }
                await shorter.close()
            }
        } catch {
            Log.error("SEARCHTEST failed", error)
        }
    }

    /// "name@from-to" (seconds): the top moment is in that file and starts inside the range.
    private static func videoMatches(_ top: VideoMomentMatch?, _ expected: String) -> (Bool, String?) {
        guard let top else { return (false, nil) }
        let got = "\(top.name)@\(String(format: "%g", Double(top.startMs) / 1000))s (\(top.scoreText))"
        let pieces = expected.components(separatedBy: "@")
        guard top.name.caseInsensitiveCompare(pieces[0]) == .orderedSame else { return (false, got) }
        guard pieces.count > 1 else { return (true, got) }
        let range = pieces[1].split(separator: "-").compactMap { Double($0) }
        guard let from = range.first else { return (false, got) }
        let to = range.count > 1 ? range[1] : from + 2
        return (Double(top.startMs) >= from * 1000 && Double(top.startMs) < to * 1000, got)
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
