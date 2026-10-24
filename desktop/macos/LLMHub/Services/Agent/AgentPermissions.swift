import Foundation

/// A one-tool MCP server (streamable HTTP, JSON responses) on the Agent bridge. Claude Code is started with
/// `--permission-prompt-tool mcp__llmhub__approve`, so every tool call that needs approval comes here and waits for the
/// user's Allow / Allow for this session / Deny in the session view. Same as core Services/Agent/AgentPermissions.cs.
@MainActor
enum AgentPermissions {
    static func handle(_ req: HttpRequest, _ out: HttpResponder, agent: AgentService) async {
        guard let request = (try? JSONSerialization.jsonObject(with: req.body)) as? [String: Any] else {
            await out.json(400, ["jsonrpc": "2.0", "id": NSNull(), "error": ["code": -32700, "message": "Parse error"]])
            return
        }
        let method = request["method"] as? String ?? ""
        guard let id = request["id"], !(id is NSNull) else {
            // Notifications (notifications/initialized and friends) need no reply.
            await out.empty(202)
            return
        }
        let params = request["params"] as? [String: Any] ?? [:]
        let result: [String: Any]
        switch method {
        case "initialize":
            result = [
                "protocolVersion": params["protocolVersion"] as? String ?? "2025-06-18",
                "capabilities": ["tools": [String: Any]()],
                "serverInfo": ["name": "llmhub", "version": "1.0"],
            ]
        case "tools/list":
            result = ["tools": [[
                "name": AgentService.permissionTool,
                "description": "Asks the LLM Hub user to approve a tool call.",
                "inputSchema": [
                    "type": "object",
                    "properties": [
                        "tool_name": ["type": "string"],
                        "input": ["type": "object"],
                        "tool_use_id": ["type": "string"],
                    ],
                    "required": ["tool_name", "input"],
                ],
            ]]]
        case "tools/call":
            let args = params["arguments"] as? [String: Any] ?? [:]
            let tool = args["tool_name"] as? String ?? ""
            let input = args["input"] as? [String: Any] ?? [:]
            let allowed = await agent.requestPermission(sessionId: req.query["session"] ?? "", tool: tool, input: input)
            let decision: [String: Any] = allowed
                ? ["behavior": "allow", "updatedInput": input]
                : ["behavior": "deny", "message": "The user denied this. Ask what to do instead."]
            result = ["content": [["type": "text", "text": AgentService.jsonString(decision)]]]
        default:
            result = [:]
        }
        await out.json(200, ["jsonrpc": "2.0", "id": id, "result": result])
    }
}
