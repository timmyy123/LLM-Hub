import React from 'react';
import {
  Plus,
  Home,
  Folder,
  Palette,
  Puzzle,
  LayoutGrid,
  Link,
  HelpCircle,
  Sparkles,
  Code2
} from 'lucide-react';

export default function IconNavRail({ activeTab, onTabChange, onNewChat }) {
  return (
    <aside className="w-14 h-full bg-[#141518] border-r border-white/10 flex flex-col items-center justify-between pt-12 pb-3 select-none z-20 shrink-0">
      {/* Top Logo & Main Nav Items */}
      <div className="flex flex-col items-center gap-4 w-full">
        {/* App Logo - Positioned below macOS traffic light buttons */}
        <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-amber-500 to-orange-600 flex items-center justify-center text-black font-bold shadow-lg shadow-amber-500/20 app-no-drag">
          <Sparkles size={18} className="text-black" />
        </div>

        <div className="w-8 h-px bg-white/10 my-1" />

        {/* Action: New Session */}
        <button
          onClick={onNewChat}
          className="p-2 rounded-xl bg-white/5 hover:bg-amber-500/20 text-slate-400 hover:text-amber-300 transition-colors app-no-drag"
          title="New Session"
        >
          <Plus size={18} />
        </button>

        {/* Navigation Items */}
        <button
          onClick={() => onTabChange('home')}
          className={`p-2.5 rounded-xl transition-all app-no-drag ${
            activeTab === 'home'
              ? 'bg-amber-500/20 text-amber-400 border border-amber-500/30'
              : 'text-slate-400 hover:text-slate-200 hover:bg-white/5'
          }`}
          title="Home Overview"
        >
          <Home size={18} />
        </button>

        <button
          onClick={() => onTabChange('studio')}
          className={`p-2.5 rounded-xl transition-all app-no-drag ${
            activeTab === 'studio' || activeTab === 'code'
              ? 'bg-amber-500/20 text-amber-400 border border-amber-500/30'
              : 'text-slate-400 hover:text-slate-200 hover:bg-white/5'
          }`}
          title="LLM Hub Studio"
        >
          <Code2 size={18} />
        </button>

        <button
          onClick={() => onTabChange('projects')}
          className={`p-2.5 rounded-xl transition-all app-no-drag ${
            activeTab === 'projects'
              ? 'bg-amber-500/20 text-amber-400 border border-amber-500/30'
              : 'text-slate-400 hover:text-slate-200 hover:bg-white/5'
          }`}
          title="Projects"
        >
          <Folder size={18} />
        </button>

        <button
          onClick={() => onTabChange('design-system')}
          className={`p-2.5 rounded-xl transition-all app-no-drag ${
            activeTab === 'design-system'
              ? 'bg-amber-500/20 text-amber-400 border border-amber-500/30'
              : 'text-slate-400 hover:text-slate-200 hover:bg-white/5'
          }`}
          title="Design Systems"
        >
          <Palette size={18} />
        </button>

        <button
          onClick={() => onTabChange('plugins')}
          className={`p-2.5 rounded-xl transition-all app-no-drag ${
            activeTab === 'plugins'
              ? 'bg-amber-500/20 text-amber-400 border border-amber-500/30'
              : 'text-slate-400 hover:text-slate-200 hover:bg-white/5'
          }`}
          title="Plugins & Skills"
        >
          <Puzzle size={18} />
        </button>

        <button
          onClick={() => onTabChange('automation')}
          className={`p-2.5 rounded-xl transition-all app-no-drag ${
            activeTab === 'automation'
              ? 'bg-amber-500/20 text-amber-400 border border-amber-500/30'
              : 'text-slate-400 hover:text-slate-200 hover:bg-white/5'
          }`}
          title="Automations"
        >
          <LayoutGrid size={18} />
        </button>

        <button
          onClick={() => onTabChange('integrations')}
          className={`p-2.5 rounded-xl transition-all app-no-drag ${
            activeTab === 'integrations'
              ? 'bg-amber-500/20 text-amber-400 border border-amber-500/30'
              : 'text-slate-400 hover:text-slate-200 hover:bg-white/5'
          }`}
          title="Integrations"
        >
          <Link size={18} />
        </button>
      </div>

      {/* Bottom Help Icon */}
      <button className="p-2.5 rounded-xl text-slate-500 hover:text-slate-300 hover:bg-white/5 transition-colors app-no-drag" title="Help & Docs">
        <HelpCircle size={18} />
      </button>
    </aside>
  );
}
