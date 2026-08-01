import React, { useState } from 'react';
import {
  Plus,
  Sliders,
  MessageSquare,
  Trash2,
  Edit2,
  Sparkles,
  Layers,
  Puzzle,
  Zap,
  Code2
} from 'lucide-react';

export default function Sidebar({
  chats,
  currentChatId,
  onNewChat,
  onSelectChat,
  onDeleteChat,
  onRenameChat,
  activeTab,
  onTabChange,
}) {
  const [editingChatId, setEditingChatId] = useState(null);
  const [editTitleInput, setEditTitleInput] = useState('');

  const handleStartRename = (e, chat) => {
    e.stopPropagation();
    setEditingChatId(chat.id);
    setEditTitleInput(chat.title);
  };

  const handleSaveRename = (e, id) => {
    e.stopPropagation();
    if (editTitleInput.trim()) {
      onRenameChat(id, editTitleInput.trim());
    }
    setEditingChatId(null);
  };

  return (
    <aside className="w-60 h-full bg-[#111318] border-r border-white/10 flex flex-col justify-between select-none text-slate-300 font-sans shrink-0">
      {/* Top Section */}
      <div className="flex-1 flex flex-col min-h-0">
        {/* New Session Button */}
        <div className="p-3">
          <button
            onClick={onNewChat}
            className="w-full flex items-center justify-center gap-2 px-3 py-2 rounded-xl bg-amber-500/10 hover:bg-amber-500/20 border border-amber-500/25 text-amber-300 text-xs font-semibold transition-all shadow-sm cursor-pointer"
          >
            <Plus size={16} />
            <span>New Session</span>
          </button>
        </div>

        {/* Studio Navigation Options */}
        <div className="px-2 py-1 space-y-0.5 text-xs text-slate-400 font-medium">
          <button
            onClick={() => onTabChange('studio')}
            className={`w-full flex items-center gap-2.5 px-3 py-1.5 rounded-lg transition-colors ${
              activeTab === 'studio' || activeTab === 'code' ? 'bg-white/10 text-white font-medium' : 'hover:bg-white/5 hover:text-slate-200'
            }`}
          >
            <Code2 size={15} className="text-amber-400" />
            <span>LLM Hub Studio</span>
          </button>
          <button
            onClick={() => onTabChange('design-system')}
            className={`w-full flex items-center gap-2.5 px-3 py-1.5 rounded-lg transition-colors ${
              activeTab === 'design-system' ? 'bg-white/10 text-white font-medium' : 'hover:bg-white/5 hover:text-slate-200'
            }`}
          >
            <Layers size={15} className="text-blue-400" />
            <span>Design Systems</span>
          </button>
          <button
            onClick={() => onTabChange('automation')}
            className={`w-full flex items-center gap-2.5 px-3 py-1.5 rounded-lg transition-colors ${
              activeTab === 'automation' ? 'bg-white/10 text-white font-medium' : 'hover:bg-white/5 hover:text-slate-200'
            }`}
          >
            <Zap size={15} className="text-purple-400" />
            <span>Automations</span>
          </button>
          <button
            onClick={() => onTabChange('plugins')}
            className={`w-full flex items-center gap-2.5 px-3 py-1.5 rounded-lg transition-colors ${
              activeTab === 'plugins' ? 'bg-white/10 text-white font-medium' : 'hover:bg-white/5 hover:text-slate-200'
            }`}
          >
            <Puzzle size={15} className="text-emerald-400" />
            <span>Plugins & Skills</span>
          </button>
        </div>

        {/* Recents Session List */}
        <div className="flex-1 flex flex-col min-h-0 mt-2 border-t border-white/5 pt-2">
          <div className="px-4 py-1.5 flex items-center justify-between text-[11px] font-medium text-slate-400">
            <span>Recent Sessions</span>
            <Sliders size={13} className="text-slate-500 hover:text-slate-300 cursor-pointer" />
          </div>

          <div className="flex-1 overflow-y-auto px-2 space-y-0.5 custom-scrollbar">
            {chats.length === 0 ? (
              <div className="px-3 py-4 text-center text-xs text-slate-500 italic">
                No recent sessions
              </div>
            ) : (
              chats.map((chat) => {
                const isActive = chat.id === currentChatId;
                const isEditing = editingChatId === chat.id;

                return (
                  <div
                    key={chat.id}
                    onClick={() => onSelectChat(chat.id)}
                    className={`group relative flex items-center justify-between px-3 py-2 rounded-lg text-xs cursor-pointer transition-all ${
                      isActive
                        ? 'bg-white/10 text-white font-medium'
                        : 'text-slate-400 hover:bg-white/5 hover:text-slate-200'
                    }`}
                  >
                    <div className="flex items-center gap-2 min-w-0 flex-1">
                      <MessageSquare size={13} className="shrink-0 text-amber-400/80" />
                      {isEditing ? (
                        <input
                          type="text"
                          value={editTitleInput}
                          onChange={(e) => setEditTitleInput(e.target.value)}
                          onBlur={(e) => handleSaveRename(e, chat.id)}
                          onKeyDown={(e) => {
                            if (e.key === 'Enter') handleSaveRename(e, chat.id);
                          }}
                          autoFocus
                          className="bg-black/50 text-white px-1.5 py-0.5 rounded text-xs outline-none w-full border border-white/20"
                        />
                      ) : (
                        <span className="truncate font-sans">{chat.title || 'Untitled Session'}</span>
                      )}
                    </div>

                    {!isEditing && (
                      <div className="opacity-0 group-hover:opacity-100 flex items-center gap-1 transition-opacity">
                        <button
                          onClick={(e) => handleStartRename(e, chat)}
                          className="p-1 rounded hover:bg-white/10 text-slate-400 hover:text-white"
                          title="Rename"
                        >
                          <Edit2 size={12} />
                        </button>
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            onDeleteChat(chat.id);
                          }}
                          className="p-1 rounded hover:bg-rose-500/20 text-slate-400 hover:text-rose-300"
                          title="Delete"
                        >
                          <Trash2 size={12} />
                        </button>
                      </div>
                    )}
                  </div>
                );
              })
            )}
          </div>
        </div>
      </div>

      {/* Footer */}
      <div className="p-3 border-t border-white/10 bg-black/30 flex items-center justify-between text-[11px] text-slate-400 font-mono">
        <div className="flex items-center gap-2">
          <Sparkles size={14} className="text-amber-400" />
          <span>LLM Hub Studio</span>
        </div>
        <span className="text-[10px] px-1.5 py-0.5 rounded bg-amber-500/20 text-amber-300 font-semibold border border-amber-500/30">
          v1.0.0
        </span>
      </div>
    </aside>
  );
}
