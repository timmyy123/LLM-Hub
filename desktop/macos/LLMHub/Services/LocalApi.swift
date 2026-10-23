import CryptoKit
import Foundation
import Network
import Observation

struct ApiRequestLog: Identifiable {
    let id = UUID()
    let time: String
    let client: String
    let path: String
    let model: String
    let status: String
}

/// OpenAI- and Anthropic-compatible HTTP API for the installed models (same endpoints as Windows).
/// GGUF models are proxied to llama-server, which speaks both APIs natively (tools included);
/// LiteRT-LM and Apple Intelligence models are translated here.
@MainActor @Observable
final class LocalApiServer {
    private(set) var isRunning = false
    private(set) var error = ""
    private(set) var requests: [ApiRequestLog] = []
    private var listener: NWListener?
    unowned let app: AppModel

    init(app: AppModel) { self.app = app }

    var baseURL: String { "http://127.0.0.1:\(AppSettings.current.apiPort)" }

    static func newKey() -> String {
        "llmhub-" + (0..<18).map { _ in String(format: "%02x", UInt8.random(in: 0...255)) }.joined()
    }

    /// The model name clients send: the catalog id when unambiguous ("gemma-4-12b"), else the full id.
    func apiName(_ m: InstalledModel) -> String {
        guard let catalogId = m.catalogId else { return m.id }
        return app.chat.models.filter { $0.catalogId == catalogId }.count > 1 ? "\(catalogId):\(m.quant.lowercased().replacingOccurrences(of: " ", with: "-"))" : catalogId
    }

    func restart() {
        listener?.cancel()
        listener = nil
        isRunning = false
        guard AppSettings.current.apiEnabled else { return }
        if AppSettings.current.apiKey.isEmpty {
            AppSettings.current.apiKey = Self.newKey()
            AppSettings.save()
        }
        do {
            let params = NWParameters.tcp
            params.allowLocalEndpointReuse = true
            let port = NWEndpoint.Port(integerLiteral: UInt16(AppSettings.current.apiPort))
            let listener: NWListener
            if AppSettings.current.apiAllowLan {
                listener = try NWListener(using: params, on: port)
            } else {
                // Loopback only. The port then comes from the endpoint; passing it again fails with EINVAL.
                params.requiredLocalEndpoint = .hostPort(host: "127.0.0.1", port: port)
                listener = try NWListener(using: params)
            }
            listener.newConnectionHandler = { [weak self] connection in
                Task { @MainActor in self?.accept(connection) }
            }
            listener.stateUpdateHandler = { [weak self] state in
                Task { @MainActor in
                    switch state {
                    case .ready: self?.isRunning = true; self?.error = ""
                    case .failed(let e):
                        Log.error("Local API failed: \(e)")
                        self?.isRunning = false
                        self?.error = L.f("ApiStartFailed", e.localizedDescription)
                    default: break
                    }
                }
            }
            listener.start(queue: .main)
            self.listener = listener
            Log.info("Local API on port \(AppSettings.current.apiPort) (LAN=\(AppSettings.current.apiAllowLan))")
        } catch {
            Log.error("Local API could not start: \(error)")
            self.error = L.f("ApiStartFailed", error.localizedDescription)
        }
    }

    // MARK: HTTP

    private func accept(_ connection: NWConnection) {
        connection.start(queue: .main)
        Task { @MainActor in
            defer { connection.cancel() }
            guard let request = await HttpRequest.read(from: connection) else { return }
            let out = HttpResponder(connection: connection)
            await self.route(request, out)
        }
    }

    private func route(_ req: HttpRequest, _ out: HttpResponder) async {
        if req.path.hasPrefix("/v1"), AppSettings.current.apiRequireKey {
            let header = req.headers["authorization"] ?? ""
            let key = header.lowercased().hasPrefix("bearer ") ? String(header.dropFirst(7)).trimmingCharacters(in: .whitespaces) : req.headers["x-api-key"] ?? ""
            guard Data(key.utf8) == Data(AppSettings.current.apiKey.utf8) else {
                await out.json(401, Self.errorBody("authentication_error", "Missing or wrong API key"))
                record(req, "", "401")
                return
            }
        }
        switch (req.method, req.path) {
        case ("GET", "/v1/models"):
            let data: [[String: Any]] = app.chat.models.map {
                ["id": apiName($0), "object": "model", "type": "model", "owned_by": "llm-hub", "display_name": $0.displayName]
            }
            await out.json(200, ["object": "list", "data": data])
            record(req, "", "200")
        case ("POST", "/v1/chat/completions"): await generate(req, out, anthropic: false)
        case ("POST", "/v1/messages"): await generate(req, out, anthropic: true)
        case ("POST", "/v1/messages/count_tokens"): await generate(req, out, anthropic: true, countOnly: true)
        case ("GET", "/"): await out.text(200, "LLM Hub local API")
        default:
            await out.json(404, Self.errorBody("not_found_error", "Unknown endpoint \(req.path)"))
        }
    }

    private func generate(_ req: HttpRequest, _ out: HttpResponder, anthropic: Bool, countOnly: Bool = false) async {
        guard var body = (try? JSONSerialization.jsonObject(with: req.body)) as? [String: Any] else {
            await out.json(400, Self.errorBody("invalid_request_error", "Body must be JSON"))
            return
        }
        let requested = body["model"] as? String ?? ""
        guard let model = app.chat.models.first(where: { apiName($0) == requested || $0.id == requested || $0.catalogId == requested }) else {
            await out.json(404, Self.errorBody("not_found_error", "Model '\(requested)' is not installed. GET /v1/models lists what is."))
            record(req, requested, "404")
            return
        }
        let stream = body["stream"] as? Bool ?? false
        do {
            try await app.engines.load(model)
            if model.format == "gguf", let base = app.engine.baseURL {
                body["model"] = model.id
                try await proxy(req, body, base, out)
            } else if countOnly {
                await out.json(200, ["input_tokens": Self.turns(body, anthropic: true).reduce(0) { $0 + $1.content.count } / 4])
            } else {
                try await translate(body, model, anthropic: anthropic, stream: stream, out)
            }
            record(req, apiName(model), stream ? L.s("ApiStreaming") : "200")
        } catch {
            Log.error("Local API request failed", error)
            await out.json(500, Self.errorBody("api_error", error.localizedDescription))
            record(req, apiName(model), "500")
        }
    }

    private func proxy(_ req: HttpRequest, _ body: [String: Any], _ base: URL, _ out: HttpResponder) async throws {
        var upstream = URLRequest(url: base.appendingPathComponent(String(req.path.dropFirst())))
        upstream.httpMethod = "POST"
        upstream.setValue("application/json", forHTTPHeaderField: "Content-Type")
        upstream.httpBody = try JSONSerialization.data(withJSONObject: body)
        upstream.timeoutInterval = 3600
        let (bytes, response) = try await URLSession.shared.bytes(for: upstream)
        let http = response as? HTTPURLResponse
        await out.head(http?.statusCode ?? 502, contentType: http?.value(forHTTPHeaderField: "Content-Type") ?? "application/json")
        var buffer = Data()
        for try await byte in bytes {
            buffer.append(byte)
            if byte == 10 || buffer.count >= 16_384 {
                await out.write(buffer)
                buffer.removeAll(keepingCapacity: true)
            }
        }
        if !buffer.isEmpty { await out.write(buffer) }
    }

    private func translate(_ body: [String: Any], _ model: InstalledModel, anthropic: Bool, stream: Bool, _ out: HttpResponder) async throws {
        let name = apiName(model)
        let id = (anthropic ? "msg_" : "chatcmpl-") + UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(24)
        let created = Int(Date().timeIntervalSince1970)
        var text = ""
        var pieces = 0
        if stream {
            await out.head(200, contentType: "text/event-stream")
            if anthropic {
                await out.sse("message_start", ["type": "message_start", "message": ["id": id, "type": "message", "role": "assistant", "model": name,
                    "content": [], "stop_reason": NSNull(), "usage": ["input_tokens": 0, "output_tokens": 0]]])
                await out.sse("content_block_start", ["type": "content_block_start", "index": 0, "content_block": ["type": "text", "text": ""]])
            }
        }
        for try await delta in app.engines.stream(model, turns: Self.turns(body, anthropic: anthropic)) {
            guard let piece = delta.content, !piece.isEmpty else { continue }
            text += piece
            pieces += 1
            guard stream else { continue }
            if anthropic {
                await out.sse("content_block_delta", ["type": "content_block_delta", "index": 0, "delta": ["type": "text_delta", "text": piece]])
            } else {
                await out.sse(nil, Self.openAiChunk(id, created, name, ["content": piece], nil))
            }
        }
        if stream && anthropic {
            await out.sse("content_block_stop", ["type": "content_block_stop", "index": 0])
            await out.sse("message_delta", ["type": "message_delta", "delta": ["stop_reason": "end_turn", "stop_sequence": NSNull()], "usage": ["output_tokens": pieces]])
            await out.sse("message_stop", ["type": "message_stop"])
        } else if stream {
            await out.sse(nil, Self.openAiChunk(id, created, name, [:], "stop"))
            await out.write(Data("data: [DONE]\n\n".utf8))
        } else if anthropic {
            await out.json(200, ["id": id, "type": "message", "role": "assistant", "model": name, "content": [["type": "text", "text": text]],
                                 "stop_reason": "end_turn", "stop_sequence": NSNull(), "usage": ["input_tokens": 0, "output_tokens": pieces]])
        } else {
            await out.json(200, ["id": id, "object": "chat.completion", "created": created, "model": name,
                                 "choices": [["index": 0, "finish_reason": "stop", "message": ["role": "assistant", "content": text]]],
                                 "usage": ["prompt_tokens": 0, "completion_tokens": pieces, "total_tokens": pieces]])
        }
    }

    /// Flattens OpenAI or Anthropic messages (string or content parts, plus Anthropic's top-level system) to text turns.
    private static func turns(_ body: [String: Any], anthropic: Bool) -> [ChatTurn] {
        func flatten(_ content: Any?) -> String {
            if let s = content as? String { return s }
            return ((content as? [[String: Any]]) ?? []).compactMap { $0["text"] as? String }.joined()
        }
        var turns: [ChatTurn] = []
        if anthropic, let system = body["system"], !flatten(system).isEmpty { turns.append(ChatTurn(role: "system", content: flatten(system))) }
        for m in body["messages"] as? [[String: Any]] ?? [] {
            turns.append(ChatTurn(role: m["role"] as? String ?? "user", content: flatten(m["content"])))
        }
        return turns
    }

    private static func openAiChunk(_ id: String, _ created: Int, _ model: String, _ delta: [String: Any], _ finish: String?) -> [String: Any] {
        ["id": id, "object": "chat.completion.chunk", "created": created, "model": model,
         "choices": [["index": 0, "delta": delta, "finish_reason": finish as Any? ?? NSNull()]]]
    }

    /// Both OpenAI ("error.message") and Anthropic ("type":"error") clients can read this shape.
    private static func errorBody(_ type: String, _ message: String) -> [String: Any] {
        ["type": "error", "error": ["type": type, "message": message]]
    }

    private func record(_ req: HttpRequest, _ model: String, _ status: String) {
        let agent = req.headers["user-agent"].flatMap { $0.split(separator: " ").first.map { String($0.split(separator: "/")[0]) } } ?? "-"
        let formatter = DateFormatter()
        formatter.dateFormat = "HH:mm:ss"
        requests.insert(ApiRequestLog(time: formatter.string(from: Date()), client: agent, path: req.path, model: model, status: status), at: 0)
        if requests.count > 50 { requests.removeLast() }
    }
}

/// A parsed HTTP/1.1 request (Content-Length bodies, which is what API clients send).
struct HttpRequest {
    let method: String
    let path: String
    let headers: [String: String]
    let body: Data

    @MainActor static func read(from connection: NWConnection) async -> HttpRequest? {
        var data = Data()
        while true {
            guard let chunk = await receive(connection) else { return nil }
            data.append(chunk)
            guard let split = data.range(of: Data("\r\n\r\n".utf8)) else {
                if data.count > 1 << 20 { return nil }
                continue
            }
            let head = String(decoding: data[..<split.lowerBound], as: UTF8.self).components(separatedBy: "\r\n")
            let parts = head.first?.split(separator: " ") ?? []
            guard parts.count >= 2 else { return nil }
            var headers: [String: String] = [:]
            for line in head.dropFirst() {
                guard let colon = line.firstIndex(of: ":") else { continue }
                headers[line[..<colon].lowercased()] = line[line.index(after: colon)...].trimmingCharacters(in: .whitespaces)
            }
            let length = Int(headers["content-length"] ?? "0") ?? 0
            var body = Data(data[split.upperBound...])
            while body.count < length {
                guard let more = await receive(connection) else { return nil }
                body.append(more)
            }
            let path = String(parts[1]).split(separator: "?").first.map(String.init) ?? "/"
            return HttpRequest(method: String(parts[0]), path: path, headers: headers, body: body)
        }
    }

    private static func receive(_ connection: NWConnection) async -> Data? {
        await withCheckedContinuation { continuation in
            connection.receive(minimumIncompleteLength: 1, maximumLength: 65_536) { data, _, complete, error in
                continuation.resume(returning: (data?.isEmpty == false) ? data : (complete || error != nil ? nil : Data()))
            }
        }
    }
}

/// Writes a response. Streams end when the connection closes (Connection: close), so no chunked encoding is needed.
@MainActor
struct HttpResponder {
    let connection: NWConnection

    func head(_ status: Int, contentType: String) async {
        let reason = HTTPURLResponse.localizedString(forStatusCode: status).capitalized
        await write(Data("HTTP/1.1 \(status) \(reason)\r\nContent-Type: \(contentType)\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n".utf8))
    }

    func json(_ status: Int, _ object: [String: Any]) async {
        let body = (try? JSONSerialization.data(withJSONObject: object)) ?? Data()
        let reason = HTTPURLResponse.localizedString(forStatusCode: status).capitalized
        await write(Data("HTTP/1.1 \(status) \(reason)\r\nContent-Type: application/json\r\nContent-Length: \(body.count)\r\nConnection: close\r\n\r\n".utf8) + body)
    }

    func text(_ status: Int, _ text: String) async {
        let body = Data(text.utf8)
        await write(Data("HTTP/1.1 \(status) OK\r\nContent-Type: text/plain\r\nContent-Length: \(body.count)\r\nConnection: close\r\n\r\n".utf8) + body)
    }

    func sse(_ event: String?, _ object: [String: Any]) async {
        let json = String(decoding: (try? JSONSerialization.data(withJSONObject: object)) ?? Data(), as: UTF8.self)
        await write(Data(((event.map { "event: \($0)\n" } ?? "") + "data: \(json)\n\n").utf8))
    }

    func write(_ data: Data) async {
        await withCheckedContinuation { continuation in
            connection.send(content: data, completion: .contentProcessed { _ in continuation.resume() })
        }
    }
}
