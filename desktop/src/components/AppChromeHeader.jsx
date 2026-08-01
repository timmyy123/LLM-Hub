import React from 'react';
import { Home, Code2, FolderOpen, Cpu, Settings, Sparkles } from 'lucide-react';

export default function AppChromeHeader({
  activeTab,
  onTabChange,
  workspacePath,
  onSelectWorkspace,
  selectedModel,
  onSelectModel,
  installedModels = [],
  onOpenModelManager,
}) {
  const availableModels = Array.isArray(installedModels)
    ? installedModels.map((m) => (typeof m === 'string' ? m : m.name || m.model || ''))
    : [];

  return (
    <header className="h-11 bg-[#121316] border-b border-white/10 flex items-center justify-between px-4 select-none app-drag-region text-xs font-sans">
      {/* Left: macOS Traffic lights spacing + Workspace Tabs */}
      <div className="flex items-center gap-3 pl-20 app-no-drag">
        <div className="flex items-center bg-black/40 p-0.5 rounded-lg border border-white/10">
          <button
            onClick={() => onTabChange('home')}
            className={`flex items-center gap-1.5 px-3 py-1 rounded-md transition-all ${
              activeTab === 'home'
                ? 'bg-white/15 text-white font-semibold shadow-sm'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Home size={13} />
            <span>Home</span>
          </button>

          <button
            onClick={() => onTabChange('studio')}
            className={`flex items-center gap-1.5 px-3 py-1 rounded-md transition-all ${
              activeTab === 'studio' || activeTab === 'code'
                ? 'bg-amber-500/20 text-amber-300 font-semibold border border-amber-500/30 shadow-sm'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Code2 size={13} />
            <span>Studio</span>
          </button>
        </div>

        {/* Working Directory Indicator */}
        <button
          onClick={onSelectWorkspace}
          className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-white/5 hover:bg-white/10 text-slate-300 border border-white/10 text-xs transition-colors max-w-xs truncate font-mono"
          title="Select Working Directory"
        >
          <FolderOpen size={13} className="text-amber-400 shrink-0" />
          <span className="truncate">
            {workspacePath ? workspacePath.split(/[\/\\]/).pop() : 'Select Working Directory'}
          </span>
        </button>
      </div>

      {/* Right: Model Switcher & Settings */}
      <div className="flex items-center gap-2 app-no-drag">
        {/* Ollama Local Models Selector */}
        <select
          value={selectedModel || ''}
          onChange={(e) => onSelectModel(e.target.value)}
          className="bg-white/10 text-slate-200 hover:bg-white/15 border border-white/10 rounded-lg px-2.5 py-1 text-xs font-mono focus:outline-none cursor-pointer"
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
          className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-white/10 hover:bg-white/15 text-slate-300 border border-white/10 font-medium transition-colors"
          title="Model Manager"
        >
          <Cpu size={13} />
          <span>Models</span>
        </button>

        <button className="p-1.5 rounded-lg bg-white/5 hover:bg-white/10 text-slate-400 hover:text-slate-200 transition-colors">
          <Settings size={14} />
        </button>
      </div>
    </header>
  );
}
