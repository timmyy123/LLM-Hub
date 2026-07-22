import React, { useState, useEffect, useRef } from 'react';
import { marked } from 'marked';
import {
  Send,
  Square,
  Sparkles,
  Paperclip,
  Terminal,
  Cpu,
  Check,
  X,
  Play,
  Zap,
  AlertCircle,
  ExternalLink,
  Copy
} from 'lucide-react';

marked.setOptions({
  gfm: true,
  breaks: true,
});

export default function AgentChat({
  messages = [],
  onSendMessage,
  isExecuting,
  onCancel,
  selectedModel,
  onSelectModel,
  installedModels = [],
  onOpenModelManager,
  workspacePath,
}) {
  const [prompt, setPrompt] = useState('');
  const [agentMode, setAgentMode] = useState('agent');
  const [executedCmds, setExecutedCmds] = useState({});
  const [copiedMessageIndex, setCopiedMessageIndex] = useState(null);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isExecuting, executedCmds]);

  const handleSubmit = (e) => {
    e?.preventDefault();
    if (!prompt.trim() || isExecuting) return;
    onSendMessage(prompt.trim(), agentMode);
    setPrompt('');
  };

  const cleanMessageContent = (rawText) => {
    if (!rawText) return '';
    return rawText
      .replace(/<<<COMMAND:\s*([^\n>]+)>>>/g, '')
      .replace(/\[Grok Agent Executing Command\]:\s*[^\n]+\n?/g, '')
      .replace(/Command executed cleanly\.\s*\n?/g, '')
      .replace(/I am physically unable to[^\n.]+\.?/gi, '')
      .replace(/There is no code or command I can generate[^\n.]+\.?/gi, '')
      .trim();
  };

  const extractCommandsFromContent = (rawText) => {
    if (!rawText) return [];
    const regex = /<<<COMMAND:\s*([^\n>]+)>>>/g;
    const commands = [];
    let match;
    while ((match = regex.exec(rawText)) !== null) {
      commands.push(match[1].trim());
    }
    return commands;
  };

  const handleCopyMessage = (text, index) => {
    navigator.clipboard.writeText(cleanMessageContent(text));
    setCopiedMessageIndex(index);
    setTimeout(() => setCopiedMessageIndex(null), 2000);
  };

  const handleApproveCommand = async (cmdStr, messageIndex) => {
    const key = `${messageIndex}_${cmdStr}`;
    setExecutedCmds((prev) => ({ ...prev, [key]: { status: 'running' } }));

    if (window.api && window.api.executeCommand) {
      const res = await window.api.executeCommand(cmdStr, workspacePath);
      const outputText = res.output || (res.success ? 'Command executed cleanly.' : res.error || 'Failed');

      setExecutedCmds((prev) => ({
        ...prev,
        [key]: {
          status: res.success ? 'approved' : 'failed',
          output: outputText,
          isServer: res.isServer,
          serverUrl: res.serverUrl,
          success: res.success,
        },
      }));

      if (!res.success || outputText.includes('EADDRINUSE') || outputText.includes('Error')) {
        onSendMessage(
          `[System Command Error for "${cmdStr}"]:\n\`\`\`\n${outputText}\n\`\`\`\nPlease analyze this error and generate a new corrected <<<COMMAND: ...>>> to fix it immediately.`,
          'agent'
        );
      }
    }
  };

  const handleRejectCommand = (cmdStr, messageIndex) => {
    const key = `${messageIndex}_${cmdStr}`;
    setExecutedCmds((prev) => ({ ...prev, [key]: { status: 'rejected' } }));
    onSendMessage(`User rejected command: "${cmdStr}"`, 'agent');
  };

  const handleSkipCommand = (cmdStr, messageIndex) => {
    const key = `${messageIndex}_${cmdStr}`;
    setExecutedCmds((prev) => ({ ...prev, [key]: { status: 'skipped' } }));
  };

  const renderMarkdown = (content) => {
    const cleaned = cleanMessageContent(content);
    if (!cleaned) return '';
    try {
      return marked.parse(cleaned);
    } catch {
      return cleaned;
    }
  };

  const availableModels = Array.isArray(installedModels)
    ? installedModels.map((m) => (typeof m === 'string' ? m : m.name || m.model || ''))
    : [];

  return (
    <div className="flex-1 flex flex-col h-full bg-[#0A0C10] text-slate-100 overflow-hidden relative font-sans select-text">
      {/* Top Header Bar */}
      <div className="h-12 border-b border-white/10 px-6 flex items-center justify-between bg-black/20 select-none">
        <div className="flex items-center gap-2">
          <span className="text-sm font-semibold text-slate-200">
            {workspacePath ? (
              <span className="flex items-center gap-1.5 text-xs text-slate-300 font-mono">
                <Terminal size={14} className="text-slate-400" />
                {workspacePath.split(/[\/\\]/).pop()}
              </span>
            ) : (
              'Workspace: No folder opened'
            )}
          </span>
        </div>

        {/* Model Selection Dropdown */}
        <div className="flex items-center gap-2">
          <select
            value={selectedModel || ''}
            onChange={(e) => onSelectModel(e.target.value)}
            className="bg-white/10 text-slate-200 hover:bg-white/15 border border-white/10 rounded-xl px-3 py-1 text-xs font-mono focus:outline-none cursor-pointer"
          >
            {availableModels.length === 0 ? (
              <option value="">No Model Installed</option>
            ) : (
              availableModels.map((m) => (
                <option key={m} value={m} className="bg-slate-900 text-slate-200">
                  {m}
                </option>
              ))
            )}
          </select>

          <button
            onClick={onOpenModelManager}
            className="px-2.5 py-1 rounded-xl bg-white/10 hover:bg-white/15 text-slate-300 text-xs font-medium border border-white/10 transition-colors flex items-center gap-1.5"
          >
            <Cpu size={13} />
            <span>Models</span>
          </button>
        </div>
      </div>

      {/* Main Content Area */}
      <div className="flex-1 overflow-y-auto px-4 md:px-8 py-6 space-y-6 custom-scrollbar select-text">
        {!messages || messages.length === 0 ? (
          /* Empty Welcome Screen */
          <div className="h-full flex flex-col items-center justify-center max-w-2xl mx-auto space-y-8 py-12 select-none">
            <div className="text-center space-y-3">
              <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-amber-500/10 border border-amber-500/20 text-amber-300 text-xs font-medium">
                <Sparkles size={14} />
                <span>Grok Build Autonomous Agent</span>
              </div>
              <h1 className="text-3xl font-serif font-medium tracking-tight text-slate-100">
                What would you like to build today?
              </h1>
              <p className="text-sm text-slate-400 max-w-md mx-auto">
                Grok Agent will automatically execute tools, generate files, and build full applications in your workspace.
              </p>
            </div>

            {/* Central Floating Prompt Box */}
            <div className="w-full liquid-glass-card rounded-2xl p-4 space-y-3 shadow-2xl border border-white/15">
              <textarea
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    handleSubmit();
                  }
                }}
                placeholder="How can I help you today? Ask to build a website, app, or execute commands..."
                rows={4}
                className="w-full bg-transparent text-sm text-slate-100 placeholder-slate-500 focus:outline-none resize-none font-sans"
              />

              <div className="flex items-center justify-between border-t border-white/10 pt-3">
                <div className="flex items-center gap-2">
                  <button className="p-2 rounded-xl text-slate-400 hover:text-slate-200 hover:bg-white/10 transition-colors">
                    <Paperclip size={16} />
                  </button>

                  <div className="flex bg-black/40 p-1 rounded-xl border border-white/5 text-xs">
                    <button
                      type="button"
                      onClick={() => setAgentMode('chat')}
                      className={`px-3 py-1 rounded-lg font-medium transition-all ${
                        agentMode === 'chat'
                          ? 'bg-white/15 text-white'
                          : 'text-slate-400 hover:text-slate-200'
                      }`}
                    >
                      Chat
                    </button>
                    <button
                      type="button"
                      onClick={() => setAgentMode('agent')}
                      className={`px-3 py-1 rounded-lg font-medium flex items-center gap-1 transition-all ${
                        agentMode === 'agent'
                          ? 'bg-amber-500/20 text-amber-300 border border-amber-500/30'
                          : 'text-slate-400 hover:text-slate-200'
                      }`}
                    >
                      <Zap size={12} />
                      Agent (Auto-Code)
                    </button>
                  </div>
                </div>

                <div className="flex items-center gap-2">
                  <button
                    onClick={handleSubmit}
                    disabled={!prompt.trim() || isExecuting}
                    className="p-2 rounded-xl bg-white text-slate-950 font-semibold hover:bg-slate-200 disabled:opacity-40 transition-colors"
                  >
                    <Send size={16} />
                  </button>
                </div>
              </div>
            </div>
          </div>
        ) : (
          /* Active Chat Stream View */
          <div className="w-full max-w-4xl mx-auto space-y-6">
            {messages.map((msg, index) => {
              const commandsInMsg = msg.role === 'assistant' ? extractCommandsFromContent(msg.content) : [];
              const cleanedText = cleanMessageContent(msg.content);

              return (
                <div
                  key={index}
                  className={`group relative flex flex-col space-y-2 w-full ${
                    msg.role === 'user' ? 'items-end' : 'items-start'
                  }`}
                >
                  <div className="flex items-center justify-between w-full px-1">
                    <span className="text-[11px] text-slate-400 font-mono">
                      {msg.role === 'user' ? 'You' : 'Grok Agent'}
                    </span>

                    {/* Copy Button on Message Hover */}
                    <button
                      onClick={() => handleCopyMessage(msg.content, index)}
                      className="opacity-0 group-hover:opacity-100 flex items-center gap-1 px-2 py-0.5 rounded bg-white/10 hover:bg-white/20 text-slate-300 text-[10px] font-mono transition-all"
                    >
                      {copiedMessageIndex === index ? (
                        <>
                          <Check size={11} className="text-emerald-400" />
                          <span>Copied</span>
                        </>
                      ) : (
                        <>
                          <Copy size={11} />
                          <span>Copy</span>
                        </>
                      )}
                    </button>
                  </div>

                  {msg.role === 'user' ? (
                    /* User Message Bubble */
                    <div className="bg-white/15 text-slate-100 border border-white/10 rounded-2xl rounded-tr-none px-4 py-2.5 max-w-xl text-sm leading-relaxed font-sans whitespace-pre-wrap select-text">
                      {msg.content}
                    </div>
                  ) : (
                    /* AI Assistant Response */
                    <div className="w-full text-slate-200 text-sm leading-relaxed font-sans break-words space-y-3 select-text">
                      {cleanedText && (
                        <div
                          className="markdown-body prose prose-invert max-w-none text-slate-200 text-sm leading-relaxed whitespace-pre-wrap break-words select-text"
                          dangerouslySetInnerHTML={{ __html: renderMarkdown(msg.content) }}
                        />
                      )}

                      {/* Cursor / Claude Code Command Execution Cards */}
                      {commandsInMsg.map((cmdStr, cmdIdx) => {
                        const cmdKey = `${index}_${cmdStr}`;
                        const cmdState = executedCmds[cmdKey];

                        return (
                          <div
                            key={cmdIdx}
                            className="mt-3 p-4 rounded-xl bg-[#131418] border border-white/15 text-xs font-mono space-y-3 shadow-xl select-text"
                          >
                            <div className="flex items-center justify-between">
                              <span className="flex items-center gap-2 font-semibold text-xs text-slate-200">
                                <Terminal size={14} className="text-amber-400" />
                                Terminal Command
                              </span>

                              {/* Status Badges */}
                              {cmdState?.status === 'approved' ? (
                                <div className="flex items-center gap-2">
                                  <span className="flex items-center gap-1 px-2.5 py-0.5 rounded-full bg-emerald-500/20 text-emerald-300 text-[11px] font-semibold border border-emerald-500/30">
                                    <Check size={12} />
                                    {cmdState.isServer ? 'Server Running' : 'Executed Successfully'}
                                  </span>
                                  {cmdState.serverUrl && (
                                    <a
                                      href={cmdState.serverUrl}
                                      target="_blank"
                                      rel="noreferrer"
                                      className="flex items-center gap-1 px-2 py-0.5 rounded-full bg-blue-500/20 text-blue-300 text-[11px] font-medium border border-blue-500/30 hover:underline"
                                    >
                                      <ExternalLink size={11} />
                                      {cmdState.serverUrl}
                                    </a>
                                  )}
                                </div>
                              ) : cmdState?.status === 'failed' ? (
                                <span className="flex items-center gap-1 px-2.5 py-0.5 rounded-full bg-rose-500/20 text-rose-300 text-[11px] font-semibold border border-rose-500/30">
                                  <AlertCircle size={12} />
                                  Command Failed
                                </span>
                              ) : cmdState?.status === 'rejected' ? (
                                <span className="flex items-center gap-1 px-2.5 py-0.5 rounded-full bg-slate-800 text-slate-400 text-[11px] font-semibold border border-slate-700">
                                  <X size={12} />
                                  Rejected
                                </span>
                              ) : cmdState?.status === 'skipped' ? (
                                <span className="px-2.5 py-0.5 rounded-full bg-slate-800 text-slate-400 text-[11px] font-semibold">
                                  Skipped
                                </span>
                              ) : cmdState?.status === 'running' ? (
                                <span className="flex items-center gap-1.5 px-2.5 py-0.5 rounded-full bg-amber-500/20 text-amber-300 text-[11px] font-semibold border border-amber-500/30 animate-pulse">
                                  <Sparkles size={12} className="animate-spin" />
                                  Running...
                                </span>
                              ) : (
                                <span className="px-2.5 py-0.5 rounded-full bg-amber-500/10 text-amber-300 text-[11px] font-semibold border border-amber-500/20">
                                  Pending Approval
                                </span>
                              )}
                            </div>

                            <div className="p-3 rounded-lg bg-black/70 border border-white/10 text-slate-100 text-xs font-mono select-text">
                              <code>{cmdStr}</code>
                            </div>

                            {/* Action Buttons */}
                            {(!cmdState || cmdState.status === 'running') && (
                              <div className="flex items-center gap-2 pt-1 font-sans select-none">
                                <button
                                  onClick={() => handleApproveCommand(cmdStr, index)}
                                  disabled={cmdState?.status === 'running'}
                                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-emerald-500 hover:bg-emerald-400 text-black font-semibold transition-colors disabled:opacity-50 text-xs"
                                >
                                  <Play size={13} />
                                  Approve & Run
                                </button>
                                <button
                                  onClick={() => handleSkipCommand(cmdStr, index)}
                                  className="px-3 py-1.5 rounded-lg bg-white/10 hover:bg-white/20 text-slate-300 transition-colors text-xs"
                                >
                                  Skip
                                </button>
                                <button
                                  onClick={() => handleRejectCommand(cmdStr, index)}
                                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-rose-500/20 hover:bg-rose-500/30 text-rose-300 border border-rose-500/30 transition-colors text-xs"
                                >
                                  <X size={13} />
                                  Reject
                                </button>
                              </div>
                            )}

                            {/* Output Terminal Console */}
                            {cmdState?.output && (
                              <div className="mt-2 p-3 rounded-lg bg-black/90 border border-white/10 text-[11px] font-mono text-slate-300 whitespace-pre-wrap max-h-56 overflow-y-auto custom-scrollbar select-text">
                                {cmdState.output}
                              </div>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>
              );
            })}

            {isExecuting && (
              <div className="flex items-center gap-3 p-4 rounded-xl bg-white/5 border border-white/10 text-xs text-slate-300 font-mono select-none">
                <Sparkles size={16} className="text-amber-400 animate-spin" />
                <span>Grok Agent processing request...</span>
                <button
                  onClick={onCancel}
                  className="ml-auto flex items-center gap-1 px-2.5 py-1 rounded-lg bg-rose-500/20 text-rose-300 hover:bg-rose-500/30 border border-rose-500/30 transition-colors"
                >
                  <Square size={12} />
                  Cancel
                </button>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {/* Sticky Bottom Prompt Console */}
      {messages && messages.length > 0 && (
        <div className="p-4 border-t border-white/10 bg-black/40 select-none">
          <div className="max-w-4xl mx-auto liquid-glass-card rounded-2xl p-3 flex items-center gap-3 border border-white/15">
            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleSubmit();
                }
              }}
              placeholder="Ask Grok Agent to modify code or generate new features..."
              rows={1}
              className="flex-1 bg-transparent text-xs text-slate-100 placeholder-slate-500 focus:outline-none resize-none font-sans"
            />

            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={() => setAgentMode(agentMode === 'agent' ? 'chat' : 'agent')}
                className={`px-2.5 py-1 text-[11px] rounded-lg font-medium transition-colors ${
                  agentMode === 'agent'
                    ? 'bg-amber-500/20 text-amber-300 border border-amber-500/30'
                    : 'bg-white/10 text-slate-400'
                }`}
              >
                {agentMode === 'agent' ? 'Agent' : 'Chat'}
              </button>

              <button
                onClick={handleSubmit}
                disabled={!prompt.trim() || isExecuting}
                className="p-1.5 rounded-lg bg-white text-slate-950 font-semibold hover:bg-slate-200 disabled:opacity-40 transition-colors"
              >
                <Send size={14} />
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
