#include "chat_messages.h"

namespace bmoe::detail {

std::vector<common_chat_msg> to_common_messages(const std::vector<ChatMessage> & messages) {
    std::vector<common_chat_msg> out;
    out.reserve(messages.size());
    for (const ChatMessage & m : messages) {
        common_chat_msg c;
        c.role = m.role;
        c.content = m.content;
        c.reasoning_content = m.reasoning;
        c.tool_call_id = m.tool_call_id;
        c.tool_name = m.name;
        c.tool_calls.reserve(m.tool_calls.size());
        for (const ChatToolCall & t : m.tool_calls) {
            common_chat_tool_call call;
            call.id = t.id;
            call.name = t.name;
            call.arguments = t.arguments;
            c.tool_calls.push_back(std::move(call));
        }
        out.push_back(std::move(c));
    }
    return out;
}

void add_tools(common_chat_templates_inputs & inputs, const std::vector<ChatTool> & tools, ToolChoice choice) {
    if (tools.empty()) return;
    inputs.tools.reserve(tools.size());
    for (const ChatTool & t : tools)
        inputs.tools.push_back({t.name, t.description, t.parameters});
    switch (choice) {
    case ToolChoice::Auto:
        inputs.tool_choice = COMMON_CHAT_TOOL_CHOICE_AUTO;
        break;
    case ToolChoice::None:
        inputs.tool_choice = COMMON_CHAT_TOOL_CHOICE_NONE;
        break;
    case ToolChoice::Required:
        inputs.tool_choice = COMMON_CHAT_TOOL_CHOICE_REQUIRED;
        break;
    }
}

std::vector<ChatToolCall> from_common_tool_calls(const std::vector<common_chat_tool_call> & calls) {
    std::vector<ChatToolCall> out;
    out.reserve(calls.size());
    for (const common_chat_tool_call & c : calls)
        out.push_back({c.id, c.name, c.arguments});
    return out;
}

} // namespace bmoe::detail
