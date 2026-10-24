import Foundation

/// One row of Settings › Local API's endpoint table: method, path and what serves it.
struct ApiEndpoint: Identifiable {
    var id: String { (anthropic ? "a" : "o") + method + path }
    let method: String
    let path: String
    let servedBy: String
    let anthropic: Bool
}

/// The OpenAI endpoints beyond chat, as listed on the design's Local API page and served by the Windows app
/// (core/LLMHub.Core/Services/LocalApiServer.Media.cs): /v1/responses (any chat model), /v1/embeddings
/// (EmbeddingGemma 2, with image and audio inputs), /v1/audio/transcriptions (Whisper), /v1/audio/speech (Supertonic) and /v1/images/generations
/// (ComfyUI). Each one uses the same engine the app itself uses for that feature.
extension LocalApiServer {
    /// Every route this server answers, in the order the design lists them.
    static var endpoints: [ApiEndpoint] {
        [
            ApiEndpoint(method: "POST", path: "/v1/chat/completions", servedBy: L.s("ApiChatFeatures"), anthropic: false),
            ApiEndpoint(method: "POST", path: "/v1/responses", servedBy: "", anthropic: false),
            ApiEndpoint(method: "POST", path: "/v1/embeddings", servedBy: "EmbeddingGemma 2", anthropic: false),
            ApiEndpoint(method: "POST", path: "/v1/audio/transcriptions", servedBy: "Whisper", anthropic: false),
            ApiEndpoint(method: "POST", path: "/v1/audio/speech", servedBy: "Supertonic", anthropic: false),
            ApiEndpoint(method: "POST", path: "/v1/images/generations", servedBy: "ComfyUI", anthropic: false),
            ApiEndpoint(method: "GET", path: "/v1/models", servedBy: "", anthropic: false),
            ApiEndpoint(method: "POST", path: "/v1/messages", servedBy: L.s("ApiChatFeatures"), anthropic: true),
            ApiEndpoint(method: "POST", path: "/v1/messages/count_tokens", servedBy: "", anthropic: true),
            ApiEndpoint(method: "GET", path: "/v1/models", servedBy: "", anthropic: true),
        ]
    }

    /// Values in minutes for the Unload after idle picker; 0 is never.
    static let unloadChoices = [10, 0, 1, 30]

    static func unloadLabel(_ minutes: Int) -> String { minutes == 0 ? L.s("ApiNever") : L.f("ApiMinutes", "\(minutes)") }

    /// OpenAI voice names mapped onto Supertonic's voices, so clients written for OpenAI keep working.
    static let voiceAliases: [String: String] = [
        "alloy": "F1", "nova": "F1", "shimmer": "F2", "coral": "F2", "sage": "F2",
        "echo": "M1", "onyx": "M1", "fable": "M2", "ash": "M2", "ballad": "M2", "verse": "M1",
    ]

    func findChatModel(_ requested: String) -> InstalledModel? {
        app.chat.models.first { apiName($0) == requested || $0.id == requested || $0.catalogId == requested }
    }

    // MARK: /v1/responses

    /// OpenAI's Responses API on any installed chat model (text in, text out; streaming as response.* events).
    func responses(_ req: HttpRequest, _ out: HttpResponder) async {
        guard let body = (try? JSONSerialization.jsonObject(with: req.body)) as? [String: Any] else {
            await out.json(400, Self.errorBody("invalid_request_error", "Body must be JSON"))
            return
        }
        let requested = body["model"] as? String ?? ""
        guard let model = findChatModel(requested) else {
            await out.json(404, Self.errorBody("not_found_error", "Model '\(requested)' is not installed. GET /v1/models lists what is."))
            record(req, requested, "404")
            return
        }
        var turns: [ChatTurn] = []
        if let instructions = body["instructions"] as? String, !instructions.isEmpty { turns.append(ChatTurn(role: "system", content: instructions)) }
        if let text = body["input"] as? String {
            turns.append(ChatTurn(role: "user", content: text))
        } else if let items = body["input"] as? [[String: Any]] {
            for item in items {
                let role = item["role"] as? String ?? "user"
                let content: String
                if let text = item["content"] as? String {
                    content = text
                } else {
                    content = (item["content"] as? [[String: Any]] ?? []).compactMap { $0["text"] as? String }.joined()
                }
                turns.append(ChatTurn(role: role == "developer" ? "system" : role, content: content))
            }
        }
        let stream = body["stream"] as? Bool ?? false
        let name = apiName(model)
        func token() -> String { String(UUID().uuidString.replacingOccurrences(of: "-", with: "").lowercased().prefix(24)) }
        let id = "resp_" + token(), messageId = "msg_" + token()
        let created = Int(Date().timeIntervalSince1970)
        func envelope(_ status: String, _ text: String) -> [String: Any] {
            ["id": id, "object": "response", "created_at": created, "status": status, "model": name,
             "output": [["type": "message", "id": messageId, "status": status, "role": "assistant",
                         "content": [["type": "output_text", "text": text, "annotations": [Any]()]]]],
             "output_text": text]
        }
        var started = false
        do {
            try await loadSerialized(model)
            markApiUse()
            if stream {
                await out.head(200, contentType: "text/event-stream")
                started = true
                await out.sse("response.created", ["type": "response.created", "response": envelope("in_progress", "")])
            }
            var text = ""
            var splitter = ThinkSplitter()
            for try await delta in app.engines.stream(model, turns: turns) {
                guard let raw = delta.content, !raw.isEmpty else { continue }
                let piece = splitter.push(raw).answer
                guard !piece.isEmpty else { continue }
                text += piece
                if stream {
                    await out.sse("response.output_text.delta", ["type": "response.output_text.delta", "item_id": messageId, "output_index": 0,
                                                                 "content_index": 0, "delta": piece])
                }
            }
            if stream {
                await out.sse("response.output_text.done", ["type": "response.output_text.done", "item_id": messageId, "output_index": 0,
                                                            "content_index": 0, "text": text])
                await out.sse("response.completed", ["type": "response.completed", "response": envelope("completed", text)])
            } else {
                await out.json(200, envelope("completed", text))
            }
            markApiUse()
            record(req, name, stream ? L.s("ApiStreaming") : "200")
        } catch {
            Log.error("Local API /v1/responses failed", error)
            if !started { await out.json(500, Self.errorBody("api_error", error.localizedDescription)) }
            record(req, name, "500")
        }
    }

    // MARK: /v1/embeddings

    /// Embeddings from an installed embedding model: text on EmbeddingGemma 2's own small llama-server, or images and
    /// audio (and text in the same space) on EmbeddingGemma 2 440M/740M through LiteRT-LM.
    func embeddings(_ req: HttpRequest, _ out: HttpResponder) async {
        guard var body = (try? JSONSerialization.jsonObject(with: req.body)) as? [String: Any] else {
            await out.json(400, Self.errorBody("invalid_request_error", "Body must be JSON"))
            return
        }
        let requested = body["model"] as? String ?? ""
        let mediaModels = app.modelStore.installed.filter { $0.isEmbedding && $0.format == "litertlm" }
            .sorted { ($0.tags.contains("audio") ? 0 : 1) < ($1.tags.contains("audio") ? 0 : 1) }
        let hasMedia = (body["input"] as? [Any])?.contains { (($0 as? [String: Any])?["type"] as? String).map { $0 == "image_url" || $0 == "input_audio" } == true } == true
        if hasMedia || mediaModels.contains(where: { apiName($0) == requested || $0.id == requested || $0.catalogId == requested }) {
            await mediaEmbeddings(req, out, body: body, requested: requested, models: mediaModels)
            return
        }
        // EmbeddingGemma 2 first when it is installed alongside the original EmbeddingGemma.
        let embedders = app.modelStore.installed.filter { $0.isEmbedding && $0.format == "gguf" }
            .sorted { ($0.catalogId == "embeddinggemma-2" ? 0 : 1) < ($1.catalogId == "embeddinggemma-2" ? 0 : 1) }
        guard let model = embedders.first(where: { apiName($0) == requested || $0.id == requested || $0.catalogId == requested }) ?? embedders.first else {
            await out.json(404, Self.errorBody("not_found_error", "No embedding model is installed. Download EmbeddingGemma 2 in Settings › Models."))
            record(req, requested, "404")
            return
        }
        let name = apiName(model)
        do {
            try await app.embeddings.load(model, contextLength: min(max(model.contextLength, 512), 8192))
            markApiUse()
            guard let base = app.embeddings.baseURL else { throw ComfyService.error("The embedding server is not running") }
            body["model"] = model.id
            var upstream = URLRequest(url: base.appendingPathComponent("v1/embeddings"))
            upstream.httpMethod = "POST"
            upstream.setValue("application/json", forHTTPHeaderField: "Content-Type")
            upstream.httpBody = try JSONSerialization.data(withJSONObject: body)
            upstream.timeoutInterval = 600
            let (data, response) = try await URLSession.shared.data(for: upstream)
            let status = (response as? HTTPURLResponse)?.statusCode ?? 502
            if var json = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] {
                json["model"] = name
                await out.json(status, json)
            } else {
                await out.data(status, contentType: "application/json", data)
            }
            record(req, name, "\(status)")
        } catch {
            Log.error("Local API /v1/embeddings failed", error)
            await out.json(500, Self.errorBody("api_error", error.localizedDescription))
            record(req, name, "500")
        }
    }

    /// /v1/embeddings with OpenAI-style content parts: strings or {"type":"text"}, {"type":"image_url","image_url":{"url":
    /// "data:image/png;base64,…"}} and {"type":"input_audio","input_audio":{"data":"…","format":"wav"}}. One vector per
    /// input item, in the same space, so a text can be compared with photos and recordings. Only data URLs: nothing is fetched.
    private func mediaEmbeddings(_ req: HttpRequest, _ out: HttpResponder, body: [String: Any], requested: String, models: [InstalledModel]) async {
        let items: [Any] = switch body["input"] {
        case let list as [Any]: list
        case let single?: [single]
        case nil: []
        }
        let needsAudio = items.contains { ($0 as? [String: Any])?["type"] as? String == "input_audio" }
        guard let model = models.first(where: { apiName($0) == requested || $0.id == requested || $0.catalogId == requested })
                ?? models.first(where: { !needsAudio || $0.tags.contains("audio") }),
              !needsAudio || model.tags.contains("audio") else {
            await out.json(404, Self.errorBody("not_found_error", needsAudio
                ? "Audio embeddings need EmbeddingGemma 2 740M. Download it in Settings › Models."
                : "Image embeddings need EmbeddingGemma 2 440M or 740M. Download one in Settings › Models."))
            record(req, requested, "404")
            return
        }
        let name = apiName(model)
        do {
            let embedder = try await app.mediaEmbedder.get(model)
            markApiUse()
            var data: [[String: Any]] = []
            for (i, item) in items.enumerated() {
                let part: EmbedPart
                switch await Self.apiPart(item) {
                case .success(let value): part = value
                case .failure(let error):
                    await out.json(400, Self.errorBody("invalid_request_error", "input[\(i)]: \(error.message)"))
                    record(req, name, "400")
                    return
                }
                guard let vector = await embedder.embed([part]) else { throw ComfyService.error("input[\(i)] could not be embedded") }
                data.append(["object": "embedding", "index": i, "embedding": vector.map(Double.init)])
            }
            await out.json(200, ["object": "list", "data": data, "model": name, "usage": ["prompt_tokens": 0, "total_tokens": 0]])
            record(req, name, "200")
        } catch {
            Log.error("Local API /v1/embeddings (media) failed", error)
            await out.json(500, Self.errorBody("api_error", error.localizedDescription))
            record(req, name, "500")
        }
    }

    struct PartError: Error {
        let message: String
    }

    /// One input item as embedder input: text as is (the client adds task prompts), photos scaled, audio as 16 kHz WAV.
    static func apiPart(_ item: Any) async -> Result<EmbedPart, PartError> {
        if let text = item as? String { return .success(.text(text)) }
        guard let part = item as? [String: Any] else { return .failure(PartError(message: "expected a string or a content part")) }
        switch part["type"] as? String {
        case "text":
            return .success(.text(part["text"] as? String ?? ""))
        case "image_url":
            let url = (part["image_url"] as? [String: Any])?["url"] as? String ?? part["image_url"] as? String ?? ""
            guard url.lowercased().hasPrefix("data:image/"), let comma = url.firstIndex(of: ",") else {
                return .failure(PartError(message: "images must be data URLs (data:image/…;base64,…)"))
            }
            let type = url[url.index(url.startIndex, offsetBy: 11)...].prefix { $0 != ";" && $0 != "," }.lowercased()
            let ext = type == "jpeg" || type == "jpg" ? "jpg" : type
            guard let bytes = Data(base64Encoded: String(url[url.index(after: comma)...]), options: .ignoreUnknownCharacters) else {
                return .failure(PartError(message: "the image is not valid base64"))
            }
            let path = FileManager.default.temporaryDirectory.appendingPathComponent("llmhub-api-\(UUID().uuidString).\(ext)")
            defer { try? FileManager.default.removeItem(at: path) }
            guard (try? bytes.write(to: path)) != nil,
                  let jpeg = await Task.detached(operation: { MediaDecoders.image(path, maxEdge: 512) }).value else {
                return .failure(PartError(message: "the image could not be read"))
            }
            return .success(.image(jpeg))
        case "input_audio":
            let audio = part["input_audio"] as? [String: Any]
            let format = (audio?["format"] as? String ?? "wav").filter { $0.isLetter || $0.isNumber }
            guard let bytes = Data(base64Encoded: audio?["data"] as? String ?? "", options: .ignoreUnknownCharacters) else {
                return .failure(PartError(message: "the audio is not valid base64"))
            }
            let path = FileManager.default.temporaryDirectory.appendingPathComponent("llmhub-api-\(UUID().uuidString).\(format)")
            defer { try? FileManager.default.removeItem(at: path) }
            guard (try? bytes.write(to: path)) != nil, let samples = await MediaDecoders.audio(path, maxSeconds: 30) else {
                return .failure(PartError(message: "the \(format) audio could not be decoded"))
            }
            return .success(.audio(MediaDecoders.wav(samples, from: 0, to: samples.count)))
        default:
            return .failure(PartError(message: "unknown content part type"))
        }
    }

    // MARK: /v1/audio/transcriptions

    /// Speech to text with Whisper (multipart upload as in OpenAI's API; json or text).
    func transcriptions(_ req: HttpRequest, _ out: HttpResponder) async {
        guard let form = Multipart.parse(req), let file = form.files["file"], !file.data.isEmpty else {
            await out.json(400, Self.errorBody("invalid_request_error", "Send the audio as multipart/form-data with a 'file' field."))
            record(req, "whisper", "400")
            return
        }
        let ext = (file.filename as NSString).pathExtension
        let path = FileManager.default.temporaryDirectory.appendingPathComponent("llmhub-api-\(UUID().uuidString)" + (ext.isEmpty ? "" : ".\(ext)"))
        defer { try? FileManager.default.removeItem(at: path) }
        do {
            try file.data.write(to: path)
            let language = form.fields["language"].flatMap { $0.isEmpty ? nil : $0 }
            let speech = app.speech
            let text = try await Self.serializedMedia { try await speech.transcribe(path, language: language) }
            if form.fields["response_format"] == "text" {
                await out.text(200, text)
            } else {
                await out.json(200, ["text": text])
            }
            record(req, "whisper", "200")
        } catch {
            Log.error("Local API /v1/audio/transcriptions failed", error)
            await out.json(500, Self.errorBody("api_error", error.localizedDescription))
            record(req, "whisper", "500")
        }
    }

    // MARK: /v1/audio/speech

    /// Text to speech with Supertonic. Returns WAV (44.1 kHz) or raw 16-bit PCM.
    func speech(_ req: HttpRequest, _ out: HttpResponder) async {
        let body = (try? JSONSerialization.jsonObject(with: req.body)) as? [String: Any] ?? [:]
        let input = (body["input"] as? String ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        let format = body["response_format"] as? String ?? "wav"
        guard !input.isEmpty, format == "wav" || format == "pcm" else {
            await out.json(400, Self.errorBody("invalid_request_error", input.isEmpty ? "'input' is empty." : "response_format must be wav or pcm."))
            record(req, "supertonic", "400")
            return
        }
        let requested = body["voice"] as? String ?? AppSettings.current.voice
        let voice = SupertonicVoices.isVoice(requested) ? requested : Self.voiceAliases[requested.lowercased()] ?? AppSettings.current.voice
        let speed = (body["speed"] as? Double).map { min(2, max(0.5, $0)) }
        do {
            let speech = app.speech
            let (url, _) = try await Self.serializedMedia { try await speech.synthesize(input, voice: voice, speed: speed) }
            defer { try? FileManager.default.removeItem(at: url) }
            let bytes = try Data(contentsOf: url)
            // PCM: the samples without the 44-byte RIFF header.
            await out.data(200, contentType: format == "pcm" ? "audio/pcm" : "audio/wav", format == "pcm" && bytes.count > 44 ? bytes.dropFirst(44) : bytes)
            record(req, "supertonic", "200")
        } catch {
            Log.error("Local API /v1/audio/speech failed", error)
            await out.json(500, Self.errorBody("api_error", error.localizedDescription))
            record(req, "supertonic", "500")
        }
    }

    // MARK: /v1/images/generations

    /// Text to image through ComfyUI (SDXL by default, Flux schnell when the model name says flux), base64 PNG.
    /// ComfyUI and the model have to be set up already (Studio installs them); a request never starts a multi-GB
    /// install on its own.
    func images(_ req: HttpRequest, _ out: HttpResponder) async {
        let body = (try? JSONSerialization.jsonObject(with: req.body)) as? [String: Any] ?? [:]
        let prompt = (body["prompt"] as? String ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        let requested = body["model"] as? String ?? ""
        let workflow = requested.lowercased().contains("flux") ? ComfyWorkflow.fluxSchnell : ComfyWorkflow.sdxl
        guard !prompt.isEmpty else {
            await out.json(400, Self.errorBody("invalid_request_error", "'prompt' is empty."))
            record(req, workflow.id, "400")
            return
        }
        let comfy = app.studio.comfy
        guard comfy.isInstalled else {
            await out.json(503, Self.errorBody("not_ready", "ComfyUI is not installed yet. Open Studio › Image in LLM Hub and generate one image to set it up, then try again."))
            record(req, workflow.id, "503")
            return
        }
        guard ComfyService.hasModels(workflow) else {
            await out.json(503, Self.errorBody("not_ready", "\(workflow.name) is not downloaded yet (\(Fmt.bytes(workflow.downloadSize))). Generate one image with it in Studio › Image, then try again."))
            record(req, workflow.id, "503")
            return
        }
        let (width, height) = Self.parseSize(body["size"] as? String)
        let count = min(4, max(1, body["n"] as? Int ?? 1))
        do {
            let files = try await Self.serializedMedia { @MainActor in
                try await comfy.ensureRunning()
                let graph = workflow.build(prompt: prompt, width: width, height: height, count: count, steps: workflow.defaultSteps,
                                           seed: Int64.random(in: 1 ..< (1 << 48)), image: nil)
                return try await comfy.run(graph) { _, _ in }
            }
            let data = files.compactMap { try? Data(contentsOf: $0) }.map { ["b64_json": $0.base64EncodedString(), "revised_prompt": prompt] }
            await out.json(200, ["created": Int(Date().timeIntervalSince1970), "data": data])
            record(req, workflow.id, "200")
        } catch {
            Log.error("Local API /v1/images/generations failed", error)
            await out.json(500, Self.errorBody("api_error", error.localizedDescription))
            record(req, workflow.id, "500")
        }
    }

    /// "1024x1024" style sizes, snapped to multiples of 64 within what the image models handle well.
    static func parseSize(_ size: String?) -> (Int, Int) {
        let parts = (size ?? "").split(separator: "x").map { Int($0.trimmingCharacters(in: .whitespaces)) }
        guard parts.count == 2, let w = parts[0], let h = parts[1] else { return (1024, 1024) }
        return (min(1536, max(512, w / 64 * 64)), min(1536, max(512, h / 64 * 64)))
    }

    // MARK: Serialization

    /// Whisper, Supertonic and ComfyUI requests run one at a time.
    private static var mediaChain: Task<Void, Never>?

    static func serializedMedia<T>(_ work: @escaping @MainActor () async throws -> T) async throws -> T {
        let previous = mediaChain
        let task = Task { @MainActor in
            await previous?.value
            return try await work()
        }
        mediaChain = Task { _ = try? await task.value }
        return try await task.value
    }

    // MARK: Unload after idle

    /// Marks that a model was loaded or used for an API client, for "Unload after idle".
    func markApiUse() {
        guard !isBridge else { return }
        Self.lastApiUse = Date()
        if Self.idleTimer == nil {
            Self.idleTimer = Timer.scheduledTimer(withTimeInterval: 30, repeats: true) { _ in
                Task { @MainActor in AppModel.shared.api.unloadIfIdle() }
            }
        }
    }

    /// Frees the models' memory once API clients have been quiet for the chosen time, unless the app itself is using
    /// them (a chat reply, an Agent turn or a design being generated).
    func unloadIfIdle(now: Date = Date()) {
        let minutes = AppSettings.current.apiUnloadAfterMinutes
        guard minutes > 0, let last = Self.lastApiUse, now.timeIntervalSince(last) >= Double(minutes) * 60 else { return }
        if app.chat.isGenerating || app.agent.sessions.contains(where: \.isWorking) || app.design.isGenerating { return }
        Self.lastApiUse = nil
        Log.info("Local API idle for \(minutes) min: unloading models")
        app.engines.stop()
        app.embeddings.stop()
    }
}

/// multipart/form-data bodies (OpenAI's audio uploads): text fields and files by field name.
enum Multipart {
    struct File {
        let filename: String
        let data: Data
    }

    static func parse(_ req: HttpRequest) -> (fields: [String: String], files: [String: File])? {
        let type = req.headers["content-type"] ?? ""
        guard type.lowercased().hasPrefix("multipart/form-data"),
              let range = type.range(of: "boundary=", options: .caseInsensitive) else { return nil }
        let boundary = type[range.upperBound...].split(separator: ";").first.map { String($0).trimmingCharacters(in: CharacterSet(charactersIn: "\" ")) } ?? ""
        guard !boundary.isEmpty else { return nil }
        let delimiter = Data(("--" + boundary).utf8)
        let body = req.body
        var fields: [String: String] = [:]
        var files: [String: File] = [:]
        var cursor = body.startIndex
        guard let first = body.range(of: delimiter, in: cursor ..< body.endIndex) else { return nil }
        cursor = first.upperBound
        while let next = body.range(of: delimiter, in: cursor ..< body.endIndex) {
            var part = body[cursor ..< next.lowerBound]
            // Each part starts after the delimiter's CRLF and ends with CRLF before the next delimiter.
            if part.starts(with: Data("\r\n".utf8)) { part = part.dropFirst(2) }
            if part.suffix(2) == Data("\r\n".utf8) { part = part.dropLast(2) }
            if let split = part.range(of: Data("\r\n\r\n".utf8)) {
                let head = String(decoding: part[part.startIndex ..< split.lowerBound], as: UTF8.self)
                let content = Data(part[split.upperBound...])
                let disposition = head.components(separatedBy: "\r\n").first { $0.lowercased().hasPrefix("content-disposition") } ?? ""
                func attribute(_ name: String) -> String? {
                    guard let r = disposition.range(of: "\(name)=\"") else { return nil }
                    let rest = disposition[r.upperBound...]
                    return rest.firstIndex(of: "\"").map { String(rest[..<$0]) }
                }
                if let name = attribute("name") {
                    if let filename = attribute("filename") {
                        files[name] = File(filename: filename, data: content)
                    } else {
                        fields[name] = String(decoding: content, as: UTF8.self)
                    }
                }
            }
            cursor = next.upperBound
            if body[cursor...].starts(with: Data("--".utf8)) { break }
        }
        return (fields, files)
    }
}
