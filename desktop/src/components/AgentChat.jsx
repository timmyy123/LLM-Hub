import React, { useState, useRef, useEffect } from 'react';
import { Send, Square, Cpu, CornerDownLeft } from 'lucide-react';
import { runGrokAgent } from '../services/grokAgentService';

export default function AgentChat({ selectedModel, workspacePath, onOpenModelManager }) {
  const [promptInput, setPromptInput] = useState('');
  const [isRunning, setIsRunning] = useState(false);
  const [messages, setMessages] = useState([
    {
      id: 'welcome',
      role: 'assistant',
      text: 'Grok Build Agent initialized with local Ollama backend.\nEnter a prompt to begin coding.',
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
                text: fullText || 'Execution complete.',
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
    <div className="flex-1 flex flex-col bg-[#0A0C10] overflow-hidden">
      {/* Subheader Bar */}
      <div className="h-11 border-b border-white/5 bg-white/5 px-6 flex items-center justify-between text-xs select-none">
        <div className="flex items-center gap-3">
          <button
            onClick={onOpenModelManager}
            className="flex items-center gap-2 text-xs text-slate-300 font-mono bg-white/10 hover:bg-white/15 px-3 py-1 rounded-lg border border-white/10 transition-colors"
          >
            <Cpu size={13} className="text-slate-300" />
            <span>Active Model: <strong className="text-white font-medium">{selectedModel}</strong></span>
          </button>
        </div>

        <div className="text-xs text-slate-400 font-mono truncate max-w-md">
          {workspacePath ? workspacePath : 'No workspace directory selected'}
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
              <span>{msg.role === 'user' ? 'You' : 'Grok Agent'}</span>
              <span>•</span>
              <span>{msg.timestamp}</span>
            </div>

            <div
              className={`max-w-3xl rounded-2xl p-4 text-sm leading-relaxed ${
                msg.role === 'user'
                  ? 'bg-white/15 text-slate-100 border border-white/20 rounded-tr-none'
                  : 'liquid-glass-card text-slate-200 rounded-tl-none font-mono'
              }`}
            >
              <pre className="whitespace-pre-wrap font-sans">{msg.text}</pre>
            </div>
          </div>
        ))}

        {/* Streaming Logs */}
        {isRunning && (
          <div className="flex flex-col items-start space-y-2">
            <div className="flex items-center gap-2 text-xs text-slate-300 font-mono">
              <span className="w-2 h-2 rounded-full bg-blue-400 animate-ping" />
              <span>Grok Agent running with {selectedModel}...</span>
            </div>
            <div className="w-full max-w-3xl bg-black/60 border border-white/10 rounded-2xl p-4 text-xs font-mono text-slate-300 space-y-1">
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

      {/* Prompt Bar */}
      <div className="p-4 bg-white/5 border-t border-white/5">
        <form onSubmit={handleSendPrompt} className="relative max-w-4xl mx-auto">
          <div className="relative flex items-center liquid-glass-input rounded-2xl transition-all">
            <textarea
              rows={2}
              value={promptInput}
              onChange={(e) => setPromptInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask Grok agent to generate code, analyze files, or perform tasks..."
              className="w-full bg-transparent px-4 py-3 text-sm text-slate-100 placeholder-slate-500 focus:outline-none resize-none font-sans"
            />

            <div className="flex items-center gap-2 pr-3">
              {isRunning ? (
                <button
                  type="button"
                  onClick={handleStopAgent}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-rose-500/20 text-rose-300 border border-rose-500/30 hover:bg-rose-500/30 text-xs font-medium transition-colors"
                >
                  <Square size={13} />
                  Stop
                </button>
              ) : (
                <button
                  type="submit"
                  disabled={!promptInput.trim()}
                  className="flex items-center gap-1.5 px-4 py-2 rounded-xl bg-white hover:bg-slate-200 text-slate-950 text-xs font-semibold transition-all disabled:opacity-40"
                >
                  <span>Run Agent</span>
                  <Send size={13} />
                </button>
              )}
            </div>
          </div>

          <div className="flex items-center justify-between mt-2 px-2 text-[11px] text-slate-500 font-mono">
            <span className="flex items-center gap-1">
              <CornerDownLeft size={11} /> Press <kbd className="px-1 py-0.5 rounded bg-white/10 text-slate-300">Cmd / Ctrl + Enter</kbd>
            </span>
            <span>Ollama Backend: http://localhost:11434</span>
          </div>
        </form>
      </div>
    </div>
  );
}
