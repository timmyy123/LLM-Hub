import React, { useState } from 'react';
import {
  Send,
  Plus,
  Sparkles,
  FolderOpen,
  Layers,
  Globe,
  Presentation,
  Layout,
  Smartphone
} from 'lucide-react';

export default function HomeView({
  onSendMessage,
  workspacePath,
  onSelectWorkspace,
}) {
  const [prompt, setPrompt] = useState('');
  const [templateType, setTemplateType] = useState('none');

  const handleSubmit = (e) => {
    e?.preventDefault();
    if (!prompt.trim()) return;
    onSendMessage(prompt.trim(), 'agent');
    setPrompt('');
  };

  const handleTemplateClick = (title, defaultPrompt) => {
    setPrompt(defaultPrompt);
  };

  return (
    <div className="flex-1 flex flex-col items-center justify-center p-8 bg-[#0A0C10] text-slate-100 overflow-y-auto custom-scrollbar font-sans select-none">
      <div className="w-full max-w-3xl space-y-8 py-8 text-center">
        {/* LLM Hub Studio Hero Title & Badge */}
        <div className="space-y-3">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-amber-500/10 border border-amber-500/20 text-amber-300 text-xs font-semibold">
            <Sparkles size={14} />
            <span>LLM Hub Studio · Claude Code Agent</span>
          </div>
          <h1 className="text-4xl md:text-5xl font-serif font-medium tracking-tight text-slate-100">
            What will you design with your agent today?
          </h1>
          <p className="text-sm text-slate-400">
            The local Claude Code Studio powered by Ollama.
          </p>
        </div>

        {/* Central Floating Prompt Box */}
        <div className="w-full liquid-glass-card rounded-2xl p-4 space-y-4 shadow-2xl border border-white/15 text-left bg-[#131418]">
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleSubmit();
              }
            }}
            placeholder="Describe what you want to build (e.g., 'Build a responsive SaaS landing page with dark mode')..."
            rows={4}
            className="w-full bg-transparent text-sm text-slate-100 placeholder-slate-500 focus:outline-none resize-none font-sans"
          />

          <div className="flex items-center justify-between border-t border-white/10 pt-3">
            <div className="flex items-center gap-2">
              <button className="p-2 rounded-xl text-slate-400 hover:text-slate-200 hover:bg-white/10 transition-colors">
                <Plus size={16} />
              </button>

              <select
                value={templateType}
                onChange={(e) => setTemplateType(e.target.value)}
                className="bg-black/50 text-slate-300 border border-white/10 rounded-xl px-3 py-1.5 text-xs font-medium outline-none cursor-pointer"
              >
                <option value="none">Template: None</option>
                <option value="website">Website Clone</option>
                <option value="deck">Slide Deck</option>
                <option value="prototype">Prototype</option>
                <option value="mobile">Mobile App</option>
              </select>
            </div>

            <div className="flex items-center gap-2">
              <button
                onClick={handleSubmit}
                disabled={!prompt.trim()}
                className="flex items-center gap-2 px-5 py-2 rounded-xl bg-orange-600 hover:bg-orange-500 text-white font-semibold text-xs shadow-lg shadow-orange-600/20 disabled:opacity-40 transition-all cursor-pointer"
              >
                <Send size={14} />
                <span>Send</span>
              </button>
            </div>
          </div>
        </div>

        {/* Design System & Working Directory Bar */}
        <div className="flex items-center justify-center gap-6 text-xs text-slate-400">
          <div className="flex items-center gap-1.5 hover:text-slate-200 cursor-pointer">
            <Layers size={14} className="text-slate-500" />
            <span>No design system</span>
          </div>

          <span className="text-slate-600">•</span>

          <button
            onClick={onSelectWorkspace}
            className="flex items-center gap-1.5 hover:text-slate-200 cursor-pointer font-mono"
          >
            <FolderOpen size={14} className="text-amber-400" />
            <span>{workspacePath ? workspacePath.split(/[\/\\]/).pop() : 'Select working directory'}</span>
          </button>
        </div>

        {/* Starter Templates Grid */}
        <div className="space-y-4 pt-4">
          <div className="text-xs text-slate-400 font-medium">Start with a template...</div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div
              onClick={() => handleTemplateClick('Website Clone', 'Build a modern responsive website clone for an AI landing page with hero, features, pricing, and dark mode.')}
              className="group p-4 rounded-2xl bg-[#131418] hover:bg-white/10 border border-white/10 text-left transition-all cursor-pointer space-y-2 shadow-lg"
            >
              <div className="w-8 h-8 rounded-xl bg-amber-500/10 border border-amber-500/20 flex items-center justify-center text-amber-400">
                <Globe size={18} />
              </div>
              <div>
                <h4 className="text-xs font-semibold text-slate-200 group-hover:text-amber-300 transition-colors">
                  Website clone
                </h4>
                <p className="text-[11px] text-slate-400 leading-tight">Source-first site reproduction</p>
              </div>
            </div>

            <div
              onClick={() => handleTemplateClick('Slide deck', 'Create a 5-slide pitch deck presentation for an AI startup in HTML/CSS slides.')}
              className="group p-4 rounded-2xl bg-[#131418] hover:bg-white/10 border border-white/10 text-left transition-all cursor-pointer space-y-2 shadow-lg"
            >
              <div className="w-8 h-8 rounded-xl bg-blue-500/10 border border-blue-500/20 flex items-center justify-center text-blue-400">
                <Presentation size={18} />
              </div>
              <div>
                <h4 className="text-xs font-semibold text-slate-200 group-hover:text-blue-300 transition-colors">
                  Slide deck
                </h4>
                <p className="text-[11px] text-slate-400 leading-tight">Presentations & pitch decks</p>
              </div>
            </div>

            <div
              onClick={() => handleTemplateClick('Prototype', 'Build an interactive web app prototype dashboard with sidebar, stats cards, and charts.')}
              className="group p-4 rounded-2xl bg-[#131418] hover:bg-white/10 border border-white/10 text-left transition-all cursor-pointer space-y-2 shadow-lg"
            >
              <div className="w-8 h-8 rounded-xl bg-purple-500/10 border border-purple-500/20 flex items-center justify-center text-purple-400">
                <Layout size={18} />
              </div>
              <div>
                <h4 className="text-xs font-semibold text-slate-200 group-hover:text-purple-300 transition-colors">
                  Prototype
                </h4>
                <p className="text-[11px] text-slate-400 leading-tight">Interactive app mockups</p>
              </div>
            </div>

            <div
              onClick={() => handleTemplateClick('Mobile app', 'Design a mobile iOS app prototype layout for a chat & social networking app.')}
              className="group p-4 rounded-2xl bg-[#131418] hover:bg-white/10 border border-white/10 text-left transition-all cursor-pointer space-y-2 shadow-lg"
            >
              <div className="w-8 h-8 rounded-xl bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center text-emerald-400">
                <Smartphone size={18} />
              </div>
              <div>
                <h4 className="text-xs font-semibold text-slate-200 group-hover:text-emerald-300 transition-colors">
                  Mobile app
                </h4>
                <p className="text-[11px] text-slate-400 leading-tight">iOS & Android mockups</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
