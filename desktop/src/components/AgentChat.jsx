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
      .replace(/\[Claude Code Agent Executing Command\]:\s*[^\n]+\n?/g, '')
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

  return (
    <div className="flex-1 flex flex-col h-full bg-[#0A0C10] text-slate-100 overflow-hidden relative font-sans select-text">
      {/* Top Console Bar (No Text Overflow!) */}
      <div className="h-10 border-b border-white/10 px-4 flex items-center justify-between bg-black/40 text-xs font-mono select-none shrink-0">
        <div className="flex items-center gap-2">
          <Terminal size={14} className="text-amber-400" />
          <span className="font-semibold text-slate-200">Claude Code Console</span>
        </div>

        <span className="text-[11px] text-slate-400">
          Model: <span className="text-amber-300 font-mono">{selectedModel || 'Local Model'}</span>
        </span>
      </div>

      {/* Main Content Area */}
      <div className="flex-1 overflow-y-auto px-4 md:px-6 py-6 space-y-6 custom-scrollbar select-text">
        {!messages || messages.length === 0 ? (
          /* Empty Chat Prompt */
          <div className="h-full flex flex-col items-center justify-center text-center p-8 space-y-4">
            <Sparkles size={32} className="text-amber-400 animate-pulse" />
            <h2 className="text-xl font-medium text-slate-200">Claude Code Agent Ready</h2>
            <p className="text-xs text-slate-400 max-w-sm">
              Type a instruction below to start generating code, building websites, or executing terminal commands.
            </p>
          </div>
        ) : (
          /* Active Chat Stream View */
          <div className="w-full max-w-full space-y-6">
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
                      {msg.role === 'user' ? 'You' : 'Claude Code Agent'}
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
                    <div className="bg-white/15 text-slate-100 border border-white/10 rounded-2xl rounded-tr-none px-4 py-2.5 max-w-md text-xs leading-relaxed font-sans whitespace-pre-wrap select-text">
                      {msg.content}
                    </div>
                  ) : (
                    /* AI Assistant Response */
                    <div className="w-full text-slate-200 text-xs leading-relaxed font-sans break-words space-y-3 select-text">
                      {cleanedText && (
                        <div
                          className="markdown-body prose prose-invert max-w-none text-slate-200 text-xs leading-relaxed whitespace-pre-wrap break-words select-text"
                          dangerouslySetInnerHTML={{ __html: renderMarkdown(msg.content) }}
                        />
                      )}

                      {/* Command Execution Cards */}
                      {commandsInMsg.map((cmdStr, cmdIdx) => {
                        const cmdKey = `${index}_${cmdStr}`;
                        const cmdState = executedCmds[cmdKey];

                        return (
                          <div
                            key={cmdIdx}
                            className="mt-3 p-3.5 rounded-xl bg-[#131418] border border-white/15 text-xs font-mono space-y-3 shadow-xl select-text"
                          >
                            <div className="flex items-center justify-between">
                              <span className="flex items-center gap-2 font-semibold text-xs text-slate-200">
                                <Terminal size={14} className="text-amber-400" />
                                Terminal Command
                              </span>

                              {/* Status Badges */}
                              {cmdState?.status === 'approved' ? (
                                <div className="flex items-center gap-2">
                                  <span className="flex items-center gap-1 px-2.5 py-0.5 rounded-full bg-emerald-500/20 text-emerald-300 text-[10px] font-semibold border border-emerald-500/30">
                                    <Check size={11} />
                                    {cmdState.isServer ? 'Server Running' : 'Executed'}
                                  </span>
                                  {cmdState.serverUrl && (
                                    <a
                                      href={cmdState.serverUrl}
                                      target="_blank"
                                      rel="noreferrer"
                                      className="flex items-center gap-1 px-2 py-0.5 rounded-full bg-blue-500/20 text-blue-300 text-[10px] font-medium border border-blue-500/30 hover:underline"
                                    >
                                      <ExternalLink size={10} />
                                      {cmdState.serverUrl}
                                    </a>
                                  )}
                                </div>
                              ) : cmdState?.status === 'failed' ? (
                                <span className="flex items-center gap-1 px-2.5 py-0.5 rounded-full bg-rose-500/20 text-rose-300 text-[10px] font-semibold border border-rose-500/30">
                                  <AlertCircle size={11} />
                                  Failed
                                </span>
                              ) : cmdState?.status === 'rejected' ? (
                                <span className="flex items-center gap-1 px-2.5 py-0.5 rounded-full bg-slate-800 text-slate-400 text-[10px] font-semibold border border-slate-700">
                                  <X size={11} />
                                  Rejected
                                </span>
                              ) : cmdState?.status === 'running' ? (
                                <span className="flex items-center gap-1.5 px-2.5 py-0.5 rounded-full bg-amber-500/20 text-amber-300 text-[10px] font-semibold border border-amber-500/30 animate-pulse">
                                  <Sparkles size={11} className="animate-spin" />
                                  Running...
                                </span>
                              ) : (
                                <span className="px-2.5 py-0.5 rounded-full bg-amber-500/10 text-amber-300 text-[10px] font-semibold border border-amber-500/20">
                                  Pending Approval
                                </span>
                              )}
                            </div>

                            <div className="p-2.5 rounded-lg bg-black/70 border border-white/10 text-slate-100 text-xs font-mono select-text">
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
                                  <Play size={12} />
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
                                  <X size={12} />
                                  Reject
                                </button>
                              </div>
                            )}

                            {/* Output Terminal Console */}
                            {cmdState?.output && (
                              <div className="mt-2 p-2.5 rounded-lg bg-black/90 border border-white/10 text-[11px] font-mono text-slate-300 whitespace-pre-wrap max-h-48 overflow-y-auto custom-scrollbar select-text">
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
              <div className="flex items-center gap-3 p-3.5 rounded-xl bg-white/5 border border-white/10 text-xs text-slate-300 font-mono select-none">
                <Sparkles size={15} className="text-amber-400 animate-spin" />
                <span>Claude Code Agent working...</span>
                <button
                  onClick={onCancel}
                  className="ml-auto flex items-center gap-1 px-2.5 py-1 rounded-lg bg-rose-500/20 text-rose-300 hover:bg-rose-500/30 border border-rose-500/30 transition-colors text-xs"
                >
                  <Square size={11} />
                  Cancel
                </button>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {/* Sticky Bottom Prompt Console */}
      <div className="p-3 border-t border-white/10 bg-black/40 select-none shrink-0">
        <div className="liquid-glass-card rounded-xl p-2.5 flex items-center gap-2 border border-white/15">
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleSubmit();
              }
            }}
            placeholder="Ask Claude Code Agent to modify code or run commands..."
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
              {agentMode === 'agent' ? 'Claude Agent' : 'Chat'}
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
    </div>
  );
}
