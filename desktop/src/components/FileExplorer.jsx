import React from 'react';
import { Folder, FileText, ChevronRight, ChevronDown, FolderOpen, HardDrive } from 'lucide-react';

export default function FileExplorer({ workspacePath, treeData, onSelectWorkspace, onSelectFile, activeFile }) {
  return (
    <div className="w-64 bg-[#14181F] border-r border-slate-800 flex flex-col h-full select-none">
      {/* Explorer Header */}
      <div className="h-12 border-b border-slate-800 px-4 flex items-center justify-between">
        <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider font-mono">Explorer</span>
        <button
          onClick={onSelectWorkspace}
          className="p-1 rounded text-slate-400 hover:text-amber-400 hover:bg-slate-800 transition-colors text-xs flex items-center gap-1 font-mono"
          title="Open Workspace Directory"
        >
          <FolderOpen size={14} />
          <span>Open</span>
        </button>
      </div>

      {/* Directory content */}
      <div className="flex-1 overflow-y-auto p-2">
        {workspacePath ? (
          <div>
            <div className="flex items-center gap-2 px-2 py-1.5 text-xs text-amber-400 font-mono font-medium truncate">
              <Folder size={14} />
              <span className="truncate">{workspacePath.split(/[\/\\]/).pop()}</span>
            </div>
            <div className="pl-2 mt-1 space-y-0.5">
              {renderTreeNodes(treeData, onSelectFile, activeFile)}
            </div>
          </div>
        ) : (
          <div className="h-full flex flex-col items-center justify-center p-6 text-center">
            <HardDrive size={32} className="text-slate-600 mb-3" />
            <p className="text-xs text-slate-400 mb-4">No directory open</p>
            <button
              onClick={onSelectWorkspace}
              className="px-3 py-1.5 bg-amber-500/10 border border-amber-500/30 hover:bg-amber-500/20 text-amber-400 text-xs font-medium rounded-lg transition-colors"
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
          <div className="flex items-center gap-1.5 px-2 py-1 rounded text-xs text-slate-400 hover:text-slate-200 hover:bg-slate-800/60 cursor-pointer">
            <ChevronRight size={13} className="text-slate-500 shrink-0" />
            <Folder size={13} className="text-amber-500/80 shrink-0" />
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
        className={`flex items-center gap-1.5 px-2 py-1 rounded text-xs cursor-pointer transition-colors ${
          isSelected
            ? 'bg-amber-500/20 text-amber-300 font-medium'
            : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
        }`}
      >
        <FileText size={13} className="text-slate-500 shrink-0" />
        <span className="truncate font-mono">{node.name}</span>
      </div>
    );
  });
}
