import Foundation
import Security

struct MCPTool: Identifiable, Equatable {
    let name: String
    let description: String
    let inputSchema: [String: Any]
    var id: String { name }
    var exposedName: String {
        "mcp_" + name.map { $0.isLetter || $0.isNumber || $0 == "_" ? String($0) : "_" }.joined()
    }

    static func == (lhs: MCPTool, rhs: MCPTool) -> Bool { lhs.name == rhs.name }

    var promptLine: String {
        let properties = inputSchema["properties"] as? [String: Any] ?? [:]
        let required = Set(inputSchema["required"] as? [String] ?? [])
        let arguments = properties.keys.sorted { lhs, rhs in
            if required.contains(lhs) != required.contains(rhs) { return required.contains(lhs) }
            return lhs < rhs
        }.map { key in
            let schema = properties[key] as? [String: Any]
            return "\"\(key)\": \(Self.argumentExample(for: schema, required: required.contains(key)))"
        }.joined(separator: ", ")
        return "- \(exposedName)({\(arguments)}): \((description.isEmpty ? name : description).prefix(300))"
    }

    private static func argumentExample(for schema: [String: Any]?, required: Bool) -> String {
        let suffix = required ? " REQUIRED" : ""
        let type = schema?["type"] as? String ?? "value"
        switch type {
        case "string":
            return "\"<string\(suffix)>\""
        case "number", "integer":
            return "\"<number\(suffix)>\""
        case "boolean":
            return required ? "true" : "false"
        case "array":
            let itemType = (schema?["items"] as? [String: Any])?["type"] as? String ?? "value"
            return itemType == "string" ? "[\"<string>\"]" : "[\"<\(itemType)>\"]"
        case "object":
            return "{}"
        default:
            return "\"<\(type)\(suffix)>\""
        }
    }
}

struct MCPSettings: Equatable {
    var enabled: Bool
    var url: String
    var token: String
}

enum MCPClientError: LocalizedError {
    case message(String)
    var errorDescription: String? {
        if case .message(let value) = self { return value }
        return nil
    }
}

@MainActor
final class MCPClient {
    static let shared = MCPClient()
    private let currentProtocol = "2026-07-28"
    private let legacyProtocol = "2025-06-18"
    private var protocolVersion = "2026-07-28"
    private var sessionID: String?
    private var requestID = 1
    private(set) var tools: [MCPTool] = []

    private var appVersion: String {
        Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "0"
    }

    private var clientInfo: [String: String] {
        ["name": "LLM Hub iOS", "version": appVersion]
    }

    func loadSettings() -> MCPSettings {
        MCPSettings(
            enabled: UserDefaults.standard.bool(forKey: "agent_mcp_enabled"),
            url: UserDefaults.standard.string(forKey: "agent_mcp_url") ?? "",
            token: MCPKeychain.read()
        )
    }

    func saveSettings(_ settings: MCPSettings) {
        UserDefaults.standard.set(settings.enabled, forKey: "agent_mcp_enabled")
        UserDefaults.standard.set(settings.url.trimmingCharacters(in: .whitespacesAndNewlines), forKey: "agent_mcp_url")
        MCPKeychain.write(settings.token.trimmingCharacters(in: .whitespacesAndNewlines))
    }

    func disconnect() {
        sessionID = nil
        protocolVersion = currentProtocol
        tools = []
    }

    func connectAndDiscover(settings: MCPSettings) async throws -> [MCPTool] {
        guard settings.enabled else { throw MCPClientError.message("MCP is disabled") }
        try validateURL(settings.url)
        disconnect()
        let result: [String: Any]
        do {
            result = try await rpc(settings: settings, method: "tools/list", params: [:], version: currentProtocol)
        } catch {
            protocolVersion = legacyProtocol
            let initialized = try await rpc(
                settings: settings,
                method: "initialize",
                params: [
                    "protocolVersion": legacyProtocol,
                    "capabilities": [:],
                    "clientInfo": clientInfo
                ],
                version: legacyProtocol,
                includeProtocolHeader: false
            )
            protocolVersion = initialized["protocolVersion"] as? String ?? legacyProtocol
            try await notify(settings: settings, method: "notifications/initialized")
            result = try await rpc(settings: settings, method: "tools/list", params: [:], version: protocolVersion)
        }
        let rawTools = result["tools"] as? [[String: Any]] ?? []
        tools = rawTools.prefix(64).compactMap { item in
            guard let name = item["name"] as? String, !name.isEmpty else { return nil }
            return MCPTool(name: name, description: item["description"] as? String ?? "", inputSchema: item["inputSchema"] as? [String: Any] ?? ["type": "object"])
        }
        return tools
    }

    func callTool(name: String, arguments: [String: Any], settings: MCPSettings) async throws -> String {
        let result = try await rpc(
            settings: settings,
            method: "tools/call",
            params: ["name": name, "arguments": arguments],
            version: protocolVersion,
            mcpName: name
        )
        var parts: [String] = []
        for item in result["content"] as? [[String: Any]] ?? [] {
            if item["type"] as? String == "text", let text = item["text"] as? String { parts.append(text) }
            else if let uri = item["uri"] as? String { parts.append(uri) }
        }
        if let structured = result["structuredContent"],
           let data = try? JSONSerialization.data(withJSONObject: structured, options: [.sortedKeys]),
           let text = String(data: data, encoding: .utf8) { parts.append(text) }
        if parts.isEmpty,
           let data = try? JSONSerialization.data(withJSONObject: result, options: [.sortedKeys]),
           let text = String(data: data, encoding: .utf8) {
            parts.append(text)
        }
        let output = String(parts.joined(separator: "\n").prefix(100_000))
        if result["isError"] as? Bool == true { throw MCPClientError.message(output.isEmpty ? "MCP tool failed" : output) }
        return output
    }

    private func validateURL(_ raw: String) throws {
        guard let url = URL(string: raw.trimmingCharacters(in: .whitespacesAndNewlines)), let scheme = url.scheme?.lowercased(), let host = url.host else {
            throw MCPClientError.message("Invalid MCP server URL")
        }
        guard scheme == "https" || (scheme == "http" && ["localhost", "127.0.0.1"].contains(host)) else {
            throw MCPClientError.message("MCP servers must use HTTPS")
        }
    }

    private func rpc(settings: MCPSettings, method: String, params: [String: Any], version: String, includeProtocolHeader: Bool = true, mcpName: String? = nil) async throws -> [String: Any] {
        let id = requestID
        requestID += 1
        var requestParams = params
        if version == currentProtocol {
            requestParams["_meta"] = ["io.modelcontextprotocol/clientInfo": clientInfo]
        }
        let response = try await execute(
            settings: settings,
            json: ["jsonrpc": "2.0", "id": id, "method": method, "params": requestParams],
            method: method,
            version: version,
            includeProtocolHeader: includeProtocolHeader,
            mcpName: mcpName
        )
        if let error = response["error"] as? [String: Any] {
            throw MCPClientError.message(error["message"] as? String ?? "MCP request failed")
        }
        guard let result = response["result"] as? [String: Any] else { throw MCPClientError.message("Invalid MCP response") }
        return result
    }

    private func notify(settings: MCPSettings, method: String) async throws {
        _ = try await execute(settings: settings, json: ["jsonrpc": "2.0", "method": method], method: method, version: protocolVersion, expectResponse: false)
    }

    private func execute(settings: MCPSettings, json: [String: Any], method: String, version: String, includeProtocolHeader: Bool = true, mcpName: String? = nil, expectResponse: Bool = true) async throws -> [String: Any] {
        guard let url = URL(string: settings.url.trimmingCharacters(in: .whitespacesAndNewlines)) else { throw MCPClientError.message("Invalid MCP server URL") }
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.timeoutInterval = 45
        request.httpBody = try JSONSerialization.data(withJSONObject: json)
        request.setValue("application/json; charset=utf-8", forHTTPHeaderField: "Content-Type")
        request.setValue("application/json, text/event-stream", forHTTPHeaderField: "Accept")
        request.setValue(method, forHTTPHeaderField: "Mcp-Method")
        if includeProtocolHeader { request.setValue(version, forHTTPHeaderField: "MCP-Protocol-Version") }
        if let mcpName { request.setValue(mcpName, forHTTPHeaderField: "Mcp-Name") }
        if let sessionID { request.setValue(sessionID, forHTTPHeaderField: "Mcp-Session-Id") }
        if !settings.token.isEmpty { request.setValue("Bearer \(settings.token)", forHTTPHeaderField: "Authorization") }
        let (data, rawResponse) = try await URLSession.shared.data(for: request)
        guard let response = rawResponse as? HTTPURLResponse else { throw MCPClientError.message("Invalid MCP response") }
        if let value = response.value(forHTTPHeaderField: "Mcp-Session-Id") { sessionID = value }
        guard (200...299).contains(response.statusCode) else {
            throw MCPClientError.message("HTTP \(response.statusCode): \(String(data: data.prefix(500), encoding: .utf8) ?? "")")
        }
        if !expectResponse || response.statusCode == 202 { return [:] }
        guard data.count <= 1_000_000 else { throw MCPClientError.message("MCP response is too large") }
        var payload = data
        if response.value(forHTTPHeaderField: "Content-Type")?.contains("text/event-stream") == true {
            let events = (String(data: data, encoding: .utf8) ?? "").split(separator: "\n").compactMap { line -> String? in
                guard line.hasPrefix("data:") else { return nil }
                return String(line.dropFirst(5)).trimmingCharacters(in: .whitespaces)
            }
            guard let event = events.last(where: { $0.contains("\"jsonrpc\"") }), let eventData = event.data(using: .utf8) else {
                throw MCPClientError.message("Empty MCP event stream")
            }
            payload = eventData
        }
        guard let object = try JSONSerialization.jsonObject(with: payload) as? [String: Any] else { throw MCPClientError.message("Invalid MCP JSON") }
        return object
    }
}

private enum MCPKeychain {
    private static let service = "com.llmhub.llmhub.mcp"
    private static let account = "bearer-token"

    static func read() -> String {
        let query: [String: Any] = [kSecClass as String: kSecClassGenericPassword, kSecAttrService as String: service, kSecAttrAccount as String: account, kSecReturnData as String: true]
        var result: CFTypeRef?
        guard SecItemCopyMatching(query as CFDictionary, &result) == errSecSuccess, let data = result as? Data else { return "" }
        return String(data: data, encoding: .utf8) ?? ""
    }

    static func write(_ value: String) {
        let query: [String: Any] = [kSecClass as String: kSecClassGenericPassword, kSecAttrService as String: service, kSecAttrAccount as String: account]
        SecItemDelete(query as CFDictionary)
        guard !value.isEmpty else { return }
        var item = query
        item[kSecValueData as String] = Data(value.utf8)
        item[kSecAttrAccessible as String] = kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly
        SecItemAdd(item as CFDictionary, nil)
    }
}
