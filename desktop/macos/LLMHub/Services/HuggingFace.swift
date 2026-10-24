import Foundation

struct CatalogSource: Decodable, Hashable {
    let repo: String
    let format: String
    let preferredQuant: String?
    /// LiteRT-LM builds that ship a multi-token-prediction drafter (faster decoding on GPU).
    let mtp: Bool?
}

struct CatalogModel: Decodable, Identifiable, Hashable {
    let id: String
    let name: String
    let maker: String
    let tags: [String]
    let contextLength: Int
    let minRamGB: Int?
    let sources: [CatalogSource]
}

struct CatalogDocument: Decodable {
    let version: Int
    let defaults: [String: String]
    let models: [CatalogModel]

    /// The curated list shared with the mobile apps (desktop/shared/catalog.json). Files come from the Hub at runtime.
    static func load() -> CatalogDocument {
        guard let url = AppPaths.bundledCatalog, let data = try? Data(contentsOf: url),
              let doc = try? JSONDecoder().decode(CatalogDocument.self, from: data) else {
            Log.error("Catalog load failed")
            return CatalogDocument(version: 1, defaults: [:], models: [])
        }
        return doc
    }
}

struct HfFile: Hashable {
    let path: String
    let size: Int64
    let sha256: String?
    var fileName: String { (path as NSString).lastPathComponent }
}

struct HfRepo: Identifiable, Hashable {
    let id: String
    let downloads: Int
    let likes: Int
    var lastModified: Date?
    /// Gated repos need the license accepted on the Hub, and a token from that account to download.
    var gated = false
    /// "gguf" or "litertlm": which library filter found it, so which files the detail pane lists.
    var format = "gguf"
    var trending = 0.0
}

/// One downloadable quantization of a GGUF repo: weight files (possibly split) plus the vision projector if any.
struct GgufVariant: Hashable {
    let repo: String
    let quant: String
    let files: [HfFile]
    let mmproj: HfFile?
    var format = "gguf"
    var weightBytes: Int64 { files.reduce(0) { $0 + $1.size } }
    var totalBytes: Int64 { weightBytes + (mmproj?.size ?? 0) }
    var modelId: String { "\(repo.replacingOccurrences(of: "/", with: "_"))__\(quant.replacingOccurrences(of: " ", with: "-"))".lowercased() }
}

enum HubError: LocalizedError {
    case status(Int)
    var errorDescription: String? {
        if case .status(let code) = self { return "HTTP \(code)" }
        return nil
    }
}

actor HuggingFaceClient {
    static let host = "https://huggingface.co"
    private var treeCache: [String: [HfFile]] = [:]

    nonisolated func request(_ url: URL) -> URLRequest {
        var request = URLRequest(url: url)
        request.setValue("LLMHub-Desktop/0.1", forHTTPHeaderField: "User-Agent")
        let token = AppSettings.current.huggingFaceToken.trimmingCharacters(in: .whitespaces)
        if !token.isEmpty, url.absoluteString.hasPrefix(Self.host) {
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }
        return request
    }

    nonisolated static func resolveURL(repo: String, path: String) -> URL {
        let escaped = path.split(separator: "/").map { $0.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? String($0) }.joined(separator: "/")
        return URL(string: "\(host)/\(repo)/resolve/main/\(escaped)")!
    }

    /// `sort`: trendingScore, downloads or lastModified. The "litert-lm" library filter marks results as .litertlm repos.
    func search(_ query: String, filter: String?, sort: String = "trendingScore") async throws -> [HfRepo] {
        var components = URLComponents(string: "\(Self.host)/api/models")!
        components.queryItems = [
            .init(name: "search", value: query), .init(name: "sort", value: sort),
            .init(name: "direction", value: "-1"), .init(name: "limit", value: "40"),
        ] + ["downloads", "likes", "lastModified", "gated", "trendingScore"].map { URLQueryItem(name: "expand[]", value: $0) }
            + (filter.map { [URLQueryItem(name: "filter", value: $0)] } ?? [])
        let (data, response) = try await URLSession.shared.data(for: request(components.url!))
        try Self.check(response)
        let items = try JSONSerialization.jsonObject(with: data) as? [[String: Any]] ?? []
        let format = filter == "litert-lm" ? "litertlm" : "gguf"
        return items.filter { !Self.isChineseOrigin($0["id"] as? String ?? "") }.map { item in
            // "gated" is false, or "auto" / "manual" for repos behind a license.
            let gated = item["gated"] is String || (item["gated"] as? Bool) == true
            return HfRepo(
                id: item["id"] as? String ?? "", downloads: item["downloads"] as? Int ?? 0, likes: item["likes"] as? Int ?? 0,
                lastModified: (item["lastModified"] as? String).flatMap(Self.parseDate), gated: gated, format: format,
                trending: (item["trendingScore"] as? NSNumber)?.doubleValue ?? 0)
        }
    }

    /// LLM Hub offers no Chinese-origin models, so search results skip them (same list as the Windows and Linux apps).
    nonisolated static func isChineseOrigin(_ repoId: String) -> Bool {
        chineseOrigin.firstMatch(in: repoId, range: NSRange(repoId.startIndex..., in: repoId)) != nil
    }

    private static let chineseOrigin = try! NSRegularExpression(
        pattern: #"(^|[/_\-. ])(qwen|qwq|deepseek|glm|chatglm|codegeex|thudm|zai-org|01-ai|yi|baichuan|internlm|internvl|intern-s1|minimax|kimi|moonshot|hunyuan|tencent|ernie|baidu|wan|wan-ai|wan2|step|stepfun|step3|seed-oss|bytedance|doubao|cogvideo|cogview|minicpm|openbmb|telechat|skywork|xverse|aquila|baai|bailing|ling|ring|inclusionai|longcat|meituan|mimo|xiaomi|xiaomimimo|kwai|kuaishou|keye|ovis|seedream|seedance|opengvlab|iflytek|spark|sensenova|pangu|hy|mistral-chinese)(?=$|[/_\-.0-9 ])"#,
        options: [.caseInsensitive])

    private nonisolated static func parseDate(_ text: String) -> Date? {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        if let date = formatter.date(from: text) { return date }
        formatter.formatOptions = [.withInternetDateTime]
        return formatter.date(from: text)
    }

    func listFiles(_ repo: String) async throws -> [HfFile] {
        if let cached = treeCache[repo] { return cached }
        let url = URL(string: "\(Self.host)/api/models/\(repo)/tree/main?recursive=true")!
        let (data, response) = try await URLSession.shared.data(for: request(url))
        try Self.check(response)
        let items = try JSONSerialization.jsonObject(with: data) as? [[String: Any]] ?? []
        let files: [HfFile] = items.compactMap { item in
            guard item["type"] as? String == "file", let path = item["path"] as? String else { return nil }
            var size = (item["size"] as? NSNumber)?.int64Value ?? 0
            var sha: String?
            if let lfs = item["lfs"] as? [String: Any] {
                sha = lfs["oid"] as? String
                size = (lfs["size"] as? NSNumber)?.int64Value ?? size
            }
            return HfFile(path: path, size: size, sha256: sha)
        }
        treeCache[repo] = files
        return files
    }

    func ggufVariants(_ repo: String) async throws -> [GgufVariant] {
        Self.groupGguf(repo: repo, files: try await listFiles(repo))
    }

    /// Generic .litertlm files. Builds made for one phone chip (Qualcomm, MediaTek, Tensor NPUs) or the web are skipped.
    func liteRtVariants(_ repo: String) async throws -> [GgufVariant] {
        let skip = ["qualcomm", "mediatek", "intel", "sm8", "mt6", "tensor", "npu", "web"]
        return try await listFiles(repo)
            .filter { f in f.path.lowercased().hasSuffix(".litertlm") && !skip.contains { f.fileName.lowercased().contains($0) } }
            .map { f in
                let name = f.fileName.lowercased()
                // Distinct label per file ("LiteRT GPU", "LiteRT int4"), so each file gets its own model id.
                let tags = ["int4", "int8", "q4", "q8", "f16", "fp16", "f32", "gpu", "cpu"].filter { name.contains($0) }
                    .map { $0 == "gpu" || $0 == "cpu" ? $0.uppercased() : $0 }
                let quant = tags.isEmpty ? "LiteRT" : "LiteRT " + tags.joined(separator: " ")
                return GgufVariant(repo: repo, quant: quant, files: [f], mmproj: nil, format: "litertlm")
            }
            .sorted { $0.weightBytes < $1.weightBytes }
    }

    private static let quantRegex = try! NSRegularExpression(
        pattern: #"(?<q>(?:UD-)?(?:IQ\d_[A-Z0-9]+|Q\d(?:_[A-Z0-9]+)*|BF16|F16|F32|MXFP4(?:_MOE)?))(?:-\d{5}-of-\d{5})?\.gguf$"#,
        options: [.caseInsensitive])

    static func groupGguf(repo: String, files: [HfFile]) -> [GgufVariant] {
        // "mtp-*.gguf" files are multi-token-prediction draft heads that ship beside the weights; they are not models.
        let all = files.filter { $0.path.lowercased().hasSuffix(".gguf") && !$0.fileName.lowercased().hasPrefix("mtp-") }
        let projectors = all.filter { $0.fileName.lowercased().contains("mmproj") }
        let mmproj = projectors.first { $0.fileName.contains("F16") && !$0.fileName.contains("BF16") }
            ?? projectors.first { $0.fileName.contains("BF16") } ?? projectors.first
        let weights = all.filter { !$0.fileName.lowercased().contains("mmproj") }
        let groups = Dictionary(grouping: weights) { file -> String in
            let name = file.fileName
            let range = NSRange(name.startIndex..., in: name)
            guard let match = quantRegex.firstMatch(in: name, range: range),
                  let r = Range(match.range(withName: "q"), in: name) else { return "GGUF" }
            return name[r].uppercased()
        }
        return groups.map { GgufVariant(repo: repo, quant: $0.key, files: $0.value.sorted { $0.path < $1.path }, mmproj: mmproj) }
            .sorted { $0.weightBytes < $1.weightBytes }
    }

    static func check(_ response: URLResponse) throws {
        if let http = response as? HTTPURLResponse, !(200..<300).contains(http.statusCode) {
            throw HubError.status(http.statusCode)
        }
    }
}
