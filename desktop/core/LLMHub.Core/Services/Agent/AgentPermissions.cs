using System.Text.Json;
using System.Text.Json.Nodes;
using Microsoft.AspNetCore.Http;

namespace LLMHub.Services.Agent;

/// <summary>
/// A one-tool MCP server (streamable HTTP, JSON responses) on the Agent bridge. Claude Code is started with
/// <c>--permission-prompt-tool mcp__llmhub__approve</c>, so every tool call that needs approval comes here and waits for the
/// user's Allow / Allow for this session / Deny in the session view.
/// </summary>
public sealed class AgentPermissions(AgentService agent)
{
    public const string ToolName = "approve";

    public async Task HandleMcpAsync(HttpContext ctx)
    {
        var request = await JsonNode.ParseAsync(ctx.Request.Body) as JsonObject;
        var method = request?["method"]?.GetValue<string>() ?? "";
        var id = request?["id"]?.DeepClone();
        if (id is null)
        {
            // Notifications (notifications/initialized and friends) need no reply.
            ctx.Response.StatusCode = 202;
            return;
        }

        JsonNode result = method switch
        {
            "initialize" => new JsonObject
            {
                ["protocolVersion"] = request?["params"]?["protocolVersion"]?.GetValue<string>() ?? "2025-06-18",
                ["capabilities"] = new JsonObject { ["tools"] = new JsonObject() },
                ["serverInfo"] = new JsonObject { ["name"] = "llmhub", ["version"] = "1.0" },
            },
            "tools/list" => new JsonObject
            {
                ["tools"] = new JsonArray(new JsonObject
                {
                    ["name"] = ToolName,
                    ["description"] = "Asks the LLM Hub user to approve a tool call.",
                    ["inputSchema"] = new JsonObject
                    {
                        ["type"] = "object",
                        ["properties"] = new JsonObject
                        {
                            ["tool_name"] = new JsonObject { ["type"] = "string" },
                            ["input"] = new JsonObject { ["type"] = "object" },
                            ["tool_use_id"] = new JsonObject { ["type"] = "string" },
                        },
                        ["required"] = new JsonArray("tool_name", "input"),
                    },
                }),
            },
            "tools/call" => await CallAsync(ctx, request!["params"] as JsonObject),
            "ping" => new JsonObject(),
            _ => new JsonObject(),
        };
        await ctx.Response.WriteAsJsonAsync(new JsonObject { ["jsonrpc"] = "2.0", ["id"] = id, ["result"] = result });
    }

    private async Task<JsonNode> CallAsync(HttpContext ctx, JsonObject? p)
    {
        var args = p?["arguments"] as JsonObject ?? new JsonObject();
        var tool = args["tool_name"]?.GetValue<string>() ?? "";
        var input = args["input"] as JsonObject ?? new JsonObject();
        var sessionId = ctx.Request.Query["session"].ToString();
        var allowed = agent.FindSession(sessionId) is { } session && await agent.RequestPermissionAsync(session, tool, input, ctx.RequestAborted);
        var decision = allowed
            ? new JsonObject { ["behavior"] = "allow", ["updatedInput"] = input.DeepClone() }
            : new JsonObject { ["behavior"] = "deny", ["message"] = "The user denied this. Ask what to do instead." };
        return new JsonObject
        {
            ["content"] = new JsonArray(new JsonObject { ["type"] = "text", ["text"] = decision.ToJsonString() }),
        };
    }
}
