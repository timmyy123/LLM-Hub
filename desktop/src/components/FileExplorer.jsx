import React from 'react';
import { Folder, FileText, ChevronRight, FolderOpen, HardDrive } from 'lucide-react';

export default function FileExplorer({ workspacePath, treeData, onSelectWorkspace, onSelectFile, activeFile }) {
  return (
    <div className="w-64 bg-white/5 border-r border-white/5 flex flex-col h-full select-none text-slate-100">
      {/* Header */}
      <div className="h-11 border-b border-white/5 px-4 flex items-center justify-between">
        <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider font-mono">Explorer</span>
        <button
          onClick={onSelectWorkspace}
          className="p-1 rounded text-slate-400 hover:text-white hover:bg-white/10 transition-colors text-xs flex items-center gap-1 font-mono"
          title="Open Workspace Directory"
        >
          <FolderOpen size={13} />
          <span>Open</span>
        </button>
      </div>

      {/* Directory content */}
      <div className="flex-1 overflow-y-auto p-2">
        {workspacePath ? (
          <div>
            <div className="flex items-center gap-2 px-2 py-1.5 text-xs text-slate-200 font-mono font-medium truncate">
              <Folder size={14} className="text-slate-400" />
              <span className="truncate">{workspacePath.split(/[\/\\]/).pop()}</span>
            </div>
            <div className="pl-2 mt-1 space-y-0.5">
              {renderTreeNodes(treeData, onSelectFile, activeFile)}
            </div>
          </div>
        ) : (
          <div className="h-full flex flex-col items-center justify-center p-6 text-center">
            <HardDrive size={28} className="text-slate-600 mb-3" />
            <p className="text-xs text-slate-400 mb-4">No folder selected</p>
            <button
              onClick={onSelectWorkspace}
              className="px-3 py-1.5 bg-white/10 hover:bg-white/15 text-slate-200 text-xs font-medium rounded-xl border border-white/10 transition-colors"
            >
              Select Folder
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

function renderTreeNodes(nodes, onSelectFile, activeFile, level = 0) {
  if (!Array.isArray(nodes)) return null;

  return nodes.map((node) => {
    const isDir = node.type === 'directory';
    const isSelected = activeFile === node.path;

    if (isDir) {
      return (
        <div key={node.path} style={{ paddingLeft: `${level * 10}px` }}>
          <div className="flex items-center gap-1.5 px-2 py-1 rounded-lg text-xs text-slate-400 hover:text-slate-200 hover:bg-white/5 cursor-pointer">
            <ChevronRight size={13} className="text-slate-500 shrink-0" />
            <Folder size={13} className="text-slate-400 shrink-0" />
            <span className="truncate font-mono">{node.name}</span>
          </div>
          {node.children && (
            <div>
              {renderTreeNodes(node.children, onSelectFile, activeFile, level + 1)}
            </div>
          )}
        </div>
      );
    }

    return (
      <div
        key={node.path}
        onClick={() => onSelectFile(node.path)}
        style={{ paddingLeft: `${(level + 1) * 10}px` }}
        className={`flex items-center gap-1.5 px-2 py-1 rounded-lg text-xs cursor-pointer transition-colors ${
          isSelected
            ? 'bg-white/15 text-white font-medium'
            : 'text-slate-400 hover:text-slate-200 hover:bg-white/5'
        }`}
      >
        <FileText size={13} className="text-slate-500 shrink-0" />
        <span className="truncate font-mono">{node.name}</span>
      </div>
    );
  });
}
