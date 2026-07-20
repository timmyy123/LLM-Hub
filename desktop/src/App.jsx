import React, { useState, useEffect } from 'react';
import FileExplorer from './components/FileExplorer';
import AgentChat from './components/AgentChat';
import ModelManagerModal from './components/ModelManagerModal';
import { filterAllowedModels } from './services/ollamaService';
import { Cpu, Wifi, WifiOff, Terminal, ShieldAlert } from 'lucide-react';

export default function App() {
  const [ollamaOnline, setOllamaOnline] = useState(false);
  const [installedModels, setInstalledModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState('gemma4:latest');
  const [workspacePath, setWorkspacePath] = useState('');
  const [workspaceTree, setWorkspaceTree] = useState([]);
  const [activeFile, setActiveFile] = useState(null);
  const [isModelModalOpen, setIsModelModalOpen] = useState(false);

  const fetchOllamaStatusAndModels = async () => {
    if (window.api && window.api.checkOllamaStatus) {
      const status = await window.api.checkOllamaStatus();
      setOllamaOnline(status.online);

      if (status.online) {
        const res = await window.api.listModels();
        if (res.success && res.models) {
          const allowed = filterAllowedModels(res.models);
          setInstalledModels(allowed);
          if (allowed.length > 0 && !allowed.some(m => m.name === selectedModel)) {
            setSelectedModel(allowed[0].name || allowed[0].model);
          }
        }
      }
    } else {
      // Browser preview mode
      setOllamaOnline(true);
      setInstalledModels([
        { name: 'gemma4:latest' },
        { name: 'ministral-3:latest' },
        { name: 'lfm2:24b-a4b' },
      ]);
    }
  };

  useEffect(() => {
    fetchOllamaStatusAndModels();
    const interval = setInterval(fetchOllamaStatusAndModels, 10000);
    return () => clearInterval(interval);
  }, []);

  const handleSelectWorkspace = async () => {
    if (window.api && window.api.selectWorkspace) {
      const path = await window.api.selectWorkspace();
      if (path) {
        setWorkspacePath(path);
        const res = await window.api.readTree(path);
        if (res.success) {
          setWorkspaceTree(res.tree);
        }
      }
    } else {
      setWorkspacePath('/Users/timmybrown/Documents/GitHub/LLM-Hub');
      setWorkspaceTree([
        { name: 'desktop', type: 'directory', children: [{ name: 'package.json', type: 'file' }] },
        { name: 'android', type: 'directory' },
        { name: 'ios', type: 'directory' },
      ]);
    }
  };

  return (
    <div className="flex flex-col h-screen w-screen bg-[#0D0F12] text-slate-100 overflow-hidden">
      {/* Top Application Bar */}
      <div className="h-10 bg-[#16191E] border-b border-slate-800/80 px-4 flex items-center justify-between text-xs select-none">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 font-semibold tracking-wide text-slate-200">
            <span className="w-2.5 h-2.5 rounded-full bg-amber-500 shadow-sm shadow-amber-500/50" />
            Grok Build Desktop
          </div>
          <span className="text-slate-600">|</span>
          <span className="text-slate-400 font-mono text-[11px]">Claude Code / Codex Agent GUI</span>
        </div>

        {/* Status Indicators */}
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-1.5 font-mono text-[11px]">
            {ollamaOnline ? (
              <span className="flex items-center gap-1.5 text-emerald-400">
                <Wifi size={13} />
                Ollama Local Active
              </span>
            ) : (
              <span className="flex items-center gap-1.5 text-red-400">
                <WifiOff size={13} />
                Ollama Offline (localhost:11434)
              </span>
            )}
          </div>

          <button
            onClick={() => setIsModelModalOpen(true)}
            className="flex items-center gap-1.5 px-2.5 py-1 rounded bg-amber-500/10 hover:bg-amber-500/20 text-amber-400 border border-amber-500/30 transition-colors font-mono text-[11px]"
          >
            <Cpu size={13} />
            <span>Models Download</span>
          </button>
        </div>
      </div>

      {/* Main Workspace Layout */}
      <div className="flex-1 flex overflow-hidden">
        <FileExplorer
          workspacePath={workspacePath}
          treeData={workspaceTree}
          onSelectWorkspace={handleSelectWorkspace}
          onSelectFile={(f) => setActiveFile(f)}
          activeFile={activeFile}
        />

        <AgentChat
          selectedModel={selectedModel}
          workspacePath={workspacePath}
          onOpenModelManager={() => setIsModelModalOpen(true)}
        />
      </div>

      {/* Local Models Manager Modal */}
      <ModelManagerModal
        isOpen={isModelModalOpen}
        onClose={() => setIsModelModalOpen(false)}
        installedModels={installedModels}
        selectedModel={selectedModel}
        onSelectModel={(tag) => {
          setSelectedModel(tag);
          setIsModelModalOpen(false);
        }}
        onRefresh={fetchOllamaStatusAndModels}
      />
    </div>
  );
}
