import React, { useState, useEffect } from 'react';
import FileExplorer from './components/FileExplorer';
import AgentChat from './components/AgentChat';
import ModelManagerModal from './components/ModelManagerModal';
import { filterAllowedModels } from './services/ollamaService';
import { Cpu, Wifi, WifiOff } from 'lucide-react';

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
      setOllamaOnline(true);
      setInstalledModels([]);
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
    <div className="flex flex-col h-screen w-screen bg-[#0A0C10] text-slate-100 overflow-hidden select-none">
      {/* macOS Liquid Glass Drag Header */}
      <div className="h-11 liquid-glass-bar px-4 flex items-center justify-between text-xs app-drag-region pl-20">
        <div className="flex items-center gap-3">
          <span className="font-medium tracking-tight text-slate-200 font-sans text-xs">
            LLM Hub Studio
          </span>
        </div>

        {/* Header Actions */}
        <div className="flex items-center gap-3 app-no-drag">
          <div className="flex items-center gap-1.5 font-mono text-[11px]">
            {ollamaOnline ? (
              <span className="flex items-center gap-1.5 text-emerald-400">
                <Wifi size={12} />
                Ollama Connected
              </span>
            ) : (
              <span className="flex items-center gap-1.5 text-rose-400">
                <WifiOff size={12} />
                Ollama Disconnected
              </span>
            )}
          </div>

          <button
            onClick={() => setIsModelModalOpen(true)}
            className="flex items-center gap-1.5 px-3 py-1 rounded-lg bg-white/10 hover:bg-white/15 text-slate-200 border border-white/10 transition-all font-sans text-[11px]"
          >
            <Cpu size={12} />
            <span>Models</span>
          </button>
        </div>
      </div>

      {/* Main Workspace */}
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

      {/* Models Manager Modal */}
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
