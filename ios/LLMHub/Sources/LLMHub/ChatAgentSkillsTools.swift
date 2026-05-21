import Foundation
import UIKit
import CryptoKit
#if canImport(LiteRTLM)
import LiteRTLM

// MARK: - ChatAgentSkillsTools

public enum ChatAgentSkillsTools {

    public static let AGENT_SYSTEM_PROMPT: String = """
You are a helpful AI assistant. You have access to the following tools:

- query_wikipedia: Look up factual summaries from Wikipedia. Use for questions about people, places, events, science, history, current affairs, or any topic that benefits from a reliable source.
- calculate_hash: Compute a cryptographic hash (MD5, SHA-1, SHA-256, SHA-512) of any text.
- send_email: Compose and open an email using the device's email app.
- send_sms: Compose and open an SMS message using the device's messaging app.
- open_map: Open a location in Apple Maps on the device.

RULES:
1. For factual questions, always use query_wikipedia first to ground your answer.
2. Use tools silently — do NOT narrate intermediate steps or tool calls.
3. After a tool returns a result, use it to compose a concise, direct final answer.
4. If no tool is relevant, answer directly from your knowledge.
5. Never output raw tool call syntax to the user.
6. IMPORTANT: When a tool result says "will be opened after" — output ONE short confirmation sentence and STOP. Example: "I've queued Maps to open Apple Park for you."
"""

    /// URL to open AFTER GPU generation completes.
    /// Opening a URL mid-stream sends the app to background and kills Metal GPU access.
    /// Tools set this; LiteRTLMBackend opens it in its defer block after generation.
    @MainActor public static var deferredOpenURL: URL? = nil

    /// Returns the array of registered Tool instances.
    public static func allTools() -> [Tool] {
        return [
            QueryWikipediaTool(),
            CalculateHashTool(),
            SendEmailTool(),
            SendSmsTool(),
            OpenMapTool()
        ]
    }
}

// MARK: - Wikipedia Tool

public struct QueryWikipediaTool: Tool {
    public static let name = "query_wikipedia"
    public static let description = "Query a summary from Wikipedia for a given topic. Use this for factual questions about people, places, events, science, history, or current affairs."

    @ToolParam(description: "Primary topic to look up (e.g., 'Albert Einstein', '2026 Oscars', 'Great Wall of China'). Extract only the key entity — remove question words and action verbs.")
    public var topic: String

    @ToolParam(description: "2-letter language code that matches the user's language (e.g., 'en', 'es', 'zh', 'fr', 'de', 'ja', 'ko', 'it', 'pt', 'ru', 'ar', 'hi').")
    public var lang: String

    public init() {}

    public func run() async throws -> Any {
        let langCode = lang.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? "en" : lang.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let encodedTopic = topic.trimmingCharacters(in: .whitespacesAndNewlines).addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) else {
            return ["error": "Invalid topic encoding", "status": "failed"]
        }
        let urlStr = "https://\(langCode).wikipedia.org/api/rest_v1/page/summary/\(encodedTopic)"
        guard let url = URL(string: urlStr) else {
            return ["error": "Invalid Wikipedia URL", "status": "failed"]
        }

        var request = URLRequest(url: url)
        request.timeoutInterval = 6.0

        do {
            let (data, response) = try await URLSession.shared.data(for: request)
            guard let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 else {
                return ["error": "No Wikipedia article found for '\(topic)'.", "status": "failed"]
            }

            if let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
               let extract = json["extract"] as? String {
                return ["result": extract, "status": "succeeded"]
            } else {
                return ["error": "No Wikipedia article found for '\(topic)'.", "status": "failed"]
            }
        } catch {
            return ["error": "Wikipedia lookup failed: \(error.localizedDescription)", "status": "failed"]
        }
    }
}

// MARK: - Hash Tool

public struct CalculateHashTool: Tool {
    public static let name = "calculate_hash"
    public static let description = "Calculate the cryptographic hash of a piece of text."

    @ToolParam(description: "The text to hash.")
    public var text: String

    @ToolParam(description: "Hash algorithm to use: MD5, SHA-1, SHA-256 (default), or SHA-512.")
    public var algorithm: String

    public init() {}

    public func run() async throws -> Any {
        let algo = algorithm.trimmingCharacters(in: .whitespacesAndNewlines).uppercased()
        let data = text.data(using: .utf8) ?? Data()

        let hashString: String
        let actualAlgo: String

        switch algo {
        case "SHA1", "SHA-1":
            let digest = Insecure.SHA1.hash(data: data)
            hashString = digest.map { String(format: "%02hhx", $0) }.joined()
            actualAlgo = "SHA-1"
        case "SHA512", "SHA-512":
            let digest = SHA512.hash(data: data)
            hashString = digest.map { String(format: "%02hhx", $0) }.joined()
            actualAlgo = "SHA-512"
        case "MD5":
            let digest = Insecure.MD5.hash(data: data)
            hashString = digest.map { String(format: "%02hhx", $0) }.joined()
            actualAlgo = "MD5"
        default:
            let digest = SHA256.hash(data: data)
            hashString = digest.map { String(format: "%02hhx", $0) }.joined()
            actualAlgo = "SHA-256"
        }

        return ["result": hashString, "algorithm": actualAlgo, "status": "succeeded"]
    }
}

// MARK: - Email Tool
// NOTE: Does NOT open the URL during GPU streaming.
// Stores it in ChatAgentSkillsTools.deferredOpenURL so the backend
// opens it AFTER generation is complete and Metal GPU is released.

public struct SendEmailTool: Tool {
    public static let name = "send_email"
    public static let description = "Send an email using the device's email app. Opens a pre-filled compose window."

    @ToolParam(description: "Recipient email address.")
    public var email: String

    @ToolParam(description: "Subject line of the email.")
    public var subject: String

    @ToolParam(description: "Body text of the email.")
    public var body: String

    public init() {}

    public func run() async throws -> Any {
        let toEmail = email.trimmingCharacters(in: .whitespacesAndNewlines)
        let subjectVal = subject
        let bodyVal = body

        guard let subjectEncoded = subjectVal.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed),
              let bodyEncoded = bodyVal.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed) else {
            return ["error": "Failed to encode email contents", "status": "failed"]
        }
        let mailtoUrlString = "mailto:\(toEmail)?subject=\(subjectEncoded)&body=\(bodyEncoded)"
        guard let url = URL(string: mailtoUrlString) else {
            return ["error": "Invalid mailto URL format", "status": "failed"]
        }

        // Defer opening until AFTER GPU generation completes (opening mid-stream
        // sends app to background and kills Metal access with permission error).
        await MainActor.run {
            ChatAgentSkillsTools.deferredOpenURL = url
        }
        return ["result": "Email to \(toEmail) will be opened after this response. Done.", "status": "succeeded"]
    }
}

// MARK: - SMS Tool
// NOTE: Same deferred-open pattern as SendEmailTool.

public struct SendSmsTool: Tool {
    public static let name = "send_sms"
    public static let description = "Send an SMS text message using the device's messaging app. Opens a pre-filled compose window."

    @ToolParam(description: "Recipient phone number (digits only, e.g., '14155552671').")
    public var phoneNumber: String

    @ToolParam(description: "Body text of the SMS message.")
    public var body: String

    public init() {}

    public func run() async throws -> Any {
        let phone = phoneNumber.trimmingCharacters(in: .whitespacesAndNewlines)
        let bodyVal = body

        guard let bodyEncoded = bodyVal.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed) else {
            return ["error": "Failed to encode SMS body", "status": "failed"]
        }
        let smsUrlString = "sms:\(phone)&body=\(bodyEncoded)"
        guard let url = URL(string: smsUrlString) else {
            return ["error": "Invalid SMS URL format", "status": "failed"]
        }

        // Defer opening until AFTER GPU generation completes.
        await MainActor.run {
            ChatAgentSkillsTools.deferredOpenURL = url
        }
        return ["result": "SMS to \(phone) will be opened after this response. Done.", "status": "succeeded"]
    }
}

// MARK: - Map Tool
// NOTE: Same deferred-open pattern. Opening maps:// mid-stream triggers
// kIOGPUCommandBufferCallbackErrorBackgroundExecutionNotPermitted.

public struct OpenMapTool: Tool {
    public static let name = "open_map"
    public static let description = "Show a location in Apple Maps on the device."

    @ToolParam(description: "Location to display (e.g., 'Eiffel Tower, Paris', 'Googleplex, Mountain View', or coordinates '37.4219983,-122.084').")
    public var location: String

    public init() {}

    public func run() async throws -> Any {
        let loc = location.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let encodedLoc = loc.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed),
              let url = URL(string: "maps://?q=\(encodedLoc)") else {
            return ["error": "Failed to encode location '\(location)'", "status": "failed"]
        }

        // Store URL — backend will open it AFTER generation is done and GPU is released.
        await MainActor.run {
            ChatAgentSkillsTools.deferredOpenURL = url
        }
        return ["result": "Apple Maps for '\(loc)' will be opened after this response. Done.", "status": "succeeded"]
    }
}

#endif
