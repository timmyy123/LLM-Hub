import React, { useState } from 'react';
import {
  Plus,
  Folder,
  FileText,
  Sliders,
  MessageSquare,
  Trash2,
  Edit2,
  Home,
  Code2,
  Cpu
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
    <aside className="w-64 h-full bg-[#131417] border-r border-white/10 flex flex-col justify-between select-none text-slate-300 font-sans">
      {/* Top Section */}
      <div className="flex-1 flex flex-col min-h-0">
        {/* macOS Traffic Lights Padding & Top Segment */}
        <div className="pt-3 px-3 pb-2 app-drag-region pl-20">
          <div className="flex bg-black/40 p-0.5 rounded-lg border border-white/5 app-no-drag">
            <button
              onClick={() => onTabChange('home')}
              className={`flex-1 py-1 text-[11px] font-medium rounded-md flex items-center justify-center gap-1.5 transition-all ${
                activeTab === 'home'
                  ? 'bg-white/15 text-white shadow-sm'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <Home size={12} />
              Home
            </button>
            <button
              onClick={() => onTabChange('code')}
              className={`flex-1 py-1 text-[11px] font-medium rounded-md flex items-center justify-center gap-1.5 transition-all ${
                activeTab === 'code'
                  ? 'bg-white/15 text-white shadow-sm'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <Code2 size={12} />
              Code
            </button>
          </div>
        </div>

        {/* New Chat Button */}
        <div className="px-3 py-2">
          <button
            onClick={onNewChat}
            className="w-full flex items-center gap-2 px-3 py-2 rounded-xl bg-white/10 hover:bg-white/15 border border-white/10 text-white text-xs font-semibold transition-all shadow-sm"
          >
            <Plus size={16} />
            <span>New chat</span>
          </button>
        </div>

        {/* Navigation Categories */}
        <div className="px-2 py-1 space-y-0.5 text-xs text-slate-400 font-medium">
          <button className="w-full flex items-center gap-2.5 px-3 py-1.5 rounded-lg hover:bg-white/5 hover:text-slate-200 transition-colors">
            <Folder size={15} />
            <span>Projects</span>
          </button>
          <button className="w-full flex items-center gap-2.5 px-3 py-1.5 rounded-lg hover:bg-white/5 hover:text-slate-200 transition-colors">
            <FileText size={15} />
            <span>Artifacts</span>
          </button>
          <button className="w-full flex items-center gap-2.5 px-3 py-1.5 rounded-lg hover:bg-white/5 hover:text-slate-200 transition-colors">
            <Sliders size={15} />
            <span>Customize</span>
          </button>
        </div>

        {/* Recents Chat List */}
        <div className="flex-1 flex flex-col min-h-0 mt-3 border-t border-white/5 pt-2">
          <div className="px-4 py-1.5 flex items-center justify-between text-[11px] font-medium text-slate-400">
            <span>Recents</span>
            <Sliders size={13} className="text-slate-500 hover:text-slate-300 cursor-pointer" />
          </div>

          <div className="flex-1 overflow-y-auto px-2 space-y-0.5 custom-scrollbar">
            {chats.length === 0 ? (
              <div className="px-3 py-4 text-center text-xs text-slate-500 italic">
                No recent chats
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
                      <MessageSquare size={13} className="shrink-0 text-slate-400" />
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
                        <span className="truncate font-sans">{chat.title || 'Untitled Chat'}</span>
                      )}
                    </div>

                    {/* Quick Options Menu */}
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

      {/* Clean System Status Footer (Zero hardcoded names or fake plans) */}
      <div className="p-3 border-t border-white/10 bg-black/20 flex items-center justify-between text-[11px] text-slate-400 font-mono">
        <div className="flex items-center gap-2">
          <Cpu size={14} className="text-slate-400" />
          <span>LLM Hub Studio</span>
        </div>
        <span className="text-[10px] px-1.5 py-0.5 rounded bg-white/10 text-slate-300 font-medium">
          v1.0.0
        </span>
      </div>
    </aside>
  );
}
