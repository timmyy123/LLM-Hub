import React, { useState, useRef, useEffect } from 'react';
import { Send, Square, Terminal, Cpu, Sparkles, Code, CheckCircle, FileText, CornerDownLeft, AlertCircle } from 'lucide-react';
import { runGrokAgent } from '../services/grokAgentService';

export default function AgentChat({ selectedModel, workspacePath, onOpenModelManager }) {
  const [promptInput, setPromptInput] = useState('');
  const [isRunning, setIsRunning] = useState(false);
  const [messages, setMessages] = useState([
    {
      id: 'welcome',
      role: 'assistant',
      text: 'Grok Build Agent initialized with local Ollama backend.\nType a task prompt to start autonomous coding.',
      timestamp: new Date().toLocaleTimeString(),
    },
  ]);
  const [activeStreamLogs, setActiveStreamLogs] = useState([]);
  const chatEndRef = useRef(null);
  const activeUnsubRef = useRef(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, activeStreamLogs]);

  const handleSendPrompt = async (e) => {
    if (e) e.preventDefault();
    if (!promptInput.trim() || isRunning) return;

    const userText = promptInput.trim();
    setPromptInput('');
    setIsRunning(true);
    setActiveStreamLogs([]);

    const userMsg = {
      id: Date.now().toString(),
      role: 'user',
      text: userText,
      timestamp: new Date().toLocaleTimeString(),
    };

    setMessages((prev) => [...prev, userMsg]);

    const res = await runGrokAgent({
      prompt: userText,
      model: selectedModel,
      workspacePath,
      onStream: (chunk) => {
        if (chunk.type === 'exit') {
          setIsRunning(false);
          setActiveStreamLogs((prevLogs) => {
            const fullText = prevLogs.map(l => l.text).join('');
            setMessages((prev) => [
              ...prev,
              {
                id: (Date.now() + 1).toString(),
                role: 'assistant',
                text: fullText || 'Grok Build task execution complete.',
                model: selectedModel,
                timestamp: new Date().toLocaleTimeString(),
              },
            ]);
            return [];
          });
        } else {
          setActiveStreamLogs((prev) => [...prev, chunk]);
        }
      },
    });

    if (res && res.unsub) {
      activeUnsubRef.current = res.unsub;
    }
  };

  const handleStopAgent = () => {
    if (activeUnsubRef.current) {
      activeUnsubRef.current();
    }
    if (window.api && window.api.cancelGrok) {
      window.api.cancelGrok();
    }
    setIsRunning(false);
  };

  const handleKeyDown = (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
      handleSendPrompt(e);
    }
  };

  return (
    <div className="flex-1 flex flex-col bg-[#0D0F12] overflow-hidden">
      {/* Agent Bar Header */}
      <div className="h-12 border-b border-slate-800 bg-[#14181F]/80 px-6 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 text-xs text-slate-300 font-mono bg-slate-900/80 px-3 py-1 rounded-lg border border-slate-800">
            <Sparkles size={14} className="text-amber-400" />
            <span>Agent: <strong className="text-amber-400 font-semibold">Grok Build</strong></span>
          </div>
          <button
            onClick={onOpenModelManager}
            className="flex items-center gap-2 text-xs text-slate-300 font-mono bg-amber-500/10 hover:bg-amber-500/20 px-3 py-1 rounded-lg border border-amber-500/30 transition-colors"
          >
            <Cpu size={14} className="text-amber-400" />
            <span>Model: <strong className="text-amber-300">{selectedModel}</strong></span>
          </button>
        </div>

        <div className="text-xs text-slate-500 font-mono truncate max-w-md">
          {workspacePath ? `Workspace: ${workspacePath}` : 'No workspace directory selected'}
        </div>
      </div>

      {/* Message Stream */}
      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        {messages.map((msg) => (
          <div
            key={msg.id}
            className={`flex flex-col ${
              msg.role === 'user' ? 'items-end' : 'items-start'
            }`}
          >
            <div className="flex items-center gap-2 text-[11px] text-slate-500 mb-1.5 font-mono">
              <span>{msg.role === 'user' ? 'You' : 'Grok Build Agent'}</span>
              <span>•</span>
              <span>{msg.timestamp}</span>
            </div>

            <div
              className={`max-w-3xl rounded-2xl p-4 text-sm leading-relaxed ${
                msg.role === 'user'
                  ? 'bg-amber-500/15 text-slate-100 border border-amber-500/30 rounded-tr-none'
                  : 'bg-[#16191E] text-slate-200 border border-slate-800 rounded-tl-none font-mono'
              }`}
            >
              <pre className="whitespace-pre-wrap font-sans">{msg.text}</pre>
            </div>
          </div>
        ))}

        {/* Live execution stream logs */}
        {isRunning && (
          <div className="flex flex-col items-start space-y-2">
            <div className="flex items-center gap-2 text-[11px] text-amber-400 font-mono animate-pulse">
              <Sparkles size={14} className="animate-spin" />
              <span>Grok Build Agent Running with {selectedModel}...</span>
            </div>
            <div className="w-full max-w-3xl bg-[#090B0D] border border-slate-800/80 rounded-2xl p-4 text-xs font-mono text-slate-300 space-y-1.5 shadow-inner">
              {activeStreamLogs.map((log, idx) => (
                <div key={idx} className="leading-snug">
                  {log.text}
                </div>
              ))}
            </div>
          </div>
        )}
        <div ref={chatEndRef} />
      </div>

      {/* Claude Code Prompt Bar */}
      <div className="p-4 bg-[#14181F]/90 border-t border-slate-800/80">
        <form onSubmit={handleSendPrompt} className="relative max-w-4xl mx-auto">
          <div className="relative flex items-center bg-[#0D0F12] border border-slate-700/80 focus-within:border-amber-500/70 rounded-2xl shadow-xl transition-all">
            <textarea
              rows={2}
              value={promptInput}
              onChange={(e) => setPromptInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask Grok Build agent to edit code, execute tasks, or analyze codebase... (Cmd + Enter)"
              className="w-full bg-transparent px-4 py-3 text-sm text-slate-100 placeholder-slate-500 focus:outline-none resize-none font-sans"
            />

            <div className="flex items-center gap-2 pr-3">
              {isRunning ? (
                <button
                  type="button"
                  onClick={handleStopAgent}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-red-500/20 text-red-400 border border-red-500/30 hover:bg-red-500/30 text-xs font-medium transition-colors"
                >
                  <Square size={13} />
                  Stop
                </button>
              ) : (
                <button
                  type="submit"
                  disabled={!promptInput.trim()}
                  className="flex items-center gap-1.5 px-4 py-2 rounded-xl bg-amber-500 hover:bg-amber-400 text-slate-950 text-xs font-semibold transition-all disabled:opacity-40 disabled:hover:bg-amber-500 shadow-md shadow-amber-500/10"
                >
                  <span>Run Agent</span>
                  <Send size={13} />
                </button>
              )}
            </div>
          </div>

          <div className="flex items-center justify-between mt-2 px-2 text-[11px] text-slate-500 font-mono">
            <span className="flex items-center gap-1">
              <CornerDownLeft size={11} /> Press <kbd className="px-1 py-0.5 rounded bg-slate-800 text-slate-300">Cmd/Ctrl + Enter</kbd> to submit
            </span>
            <span>Local Backend: Ollama (http://localhost:11434)</span>
          </div>
        </form>
      </div>
    </div>
  );
}
