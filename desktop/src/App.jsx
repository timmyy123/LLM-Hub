import React, { useState, useEffect } from 'react';
import IconNavRail from './components/IconNavRail';
import Sidebar from './components/Sidebar';
import AppChromeHeader from './components/AppChromeHeader';
import HomeView from './components/HomeView';
import FileExplorer from './components/FileExplorer';
import AgentChat from './components/AgentChat';
import CodeEditor from './components/CodeEditor';
import ArtifactPreview from './components/ArtifactPreview';
import ModelManagerModal from './components/ModelManagerModal';
import { filterAllowedModels } from './services/ollamaService';
import { Layers, Zap, Puzzle, Link as LinkIcon } from 'lucide-react';

export default function App() {
  const [ollamaOnline, setOllamaOnline] = useState(false);
  const [installedModels, setInstalledModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState(null);
  const [workspacePath, setWorkspacePath] = useState('');
  const [workspaceTree, setWorkspaceTree] = useState([]);
  const [activeFile, setActiveFile] = useState(null);
  const [isModelModalOpen, setIsModelModalOpen] = useState(false);

  // Persistent Chats & Navigation State
  const [chats, setChats] = useState([]);
  const [currentChatId, setCurrentChatId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [isExecuting, setIsExecuting] = useState(false);
  const [activeTab, setActiveTab] = useState('home'); // 'home' | 'studio' | 'projects' | 'design-system' | 'automation' | 'plugins' | 'integrations'
  const [studioRightPane, setStudioRightPane] = useState('editor'); // 'editor' | 'preview'

  // Load Installed Models
  const fetchOllamaStatusAndModels = async () => {
    if (window.api && window.api.checkOllamaStatus) {
      const status = await window.api.checkOllamaStatus();
      setOllamaOnline(status.online);

      if (status.online) {
        const res = await window.api.listModels();
        if (res.success && res.models) {
          const allowed = filterAllowedModels(res.models);
          setInstalledModels(allowed);
          if (allowed.length > 0) {
            setSelectedModel((prev) => prev || allowed[0].name || allowed[0].model);
          } else {
            setSelectedModel(null);
          }
        }
      }
    }
  };

  // Load Persistent Chats from Storage
  const loadChatsFromStorage = async () => {
    if (window.api && window.api.listChats) {
      const res = await window.api.listChats();
      if (res.success && res.chats) {
        setChats(res.chats);
        if (res.chats.length > 0 && !currentChatId) {
          setCurrentChatId(res.chats[0].id);
          setMessages(res.chats[0].messages || []);
        }
      }
    }
  };

  useEffect(() => {
    fetchOllamaStatusAndModels();
    loadChatsFromStorage();
    const interval = setInterval(fetchOllamaStatusAndModels, 5000);
    return () => clearInterval(interval);
  }, []);

  // Listen for agent streaming chunks
  useEffect(() => {
    if (window.api && window.api.onGrokStream) {
      const unsub = window.api.onGrokStream((data) => {
        if (data.type === 'stdout' && data.text) {
          setMessages((prev) => {
            const newArr = [...prev];
            if (newArr.length > 0 && newArr[newArr.length - 1].role === 'assistant') {
              newArr[newArr.length - 1] = {
                ...newArr[newArr.length - 1],
                content: newArr[newArr.length - 1].content + data.text,
              };
            } else {
              newArr.push({ role: 'assistant', content: data.text });
            }
            return newArr;
          });

          // Refresh workspace tree if agent generated files
          if (workspacePath) {
            window.api.readTree(workspacePath).then((res) => {
              if (res.success) setWorkspaceTree(res.tree);
            });
          }
        } else if (data.type === 'exit') {
          setIsExecuting(false);
          setMessages((latestMessages) => {
            saveCurrentChat(latestMessages);
            return latestMessages;
          });
        }
      });
      return () => unsub();
    }
  }, [workspacePath, currentChatId]);

  const saveCurrentChat = async (currentMessages) => {
    if (!currentChatId || currentMessages.length === 0) return;
    const title = currentMessages[0]?.content?.slice(0, 32) || 'New Session';
    const chatObject = {
      id: currentChatId,
      title,
      messages: currentMessages,
      workspacePath,
      model: selectedModel,
    };

    if (window.api && window.api.saveChat) {
      await window.api.saveChat(chatObject);
      loadChatsFromStorage();
    }
  };

  const handleNewChat = () => {
    const newId = `session_${Date.now()}`;
    setCurrentChatId(newId);
    setMessages([]);
    setActiveTab('studio');
  };

  const handleSelectChat = (id) => {
    const target = chats.find((c) => c.id === id);
    if (target) {
      setCurrentChatId(id);
      setMessages(target.messages || []);
      if (target.workspacePath) {
        setWorkspacePath(target.workspacePath);
        window.api.readTree(target.workspacePath).then((res) => {
          if (res.success) setWorkspaceTree(res.tree);
        });
      }
      setActiveTab('studio');
    }
  };

  const handleDeleteChat = async (id) => {
    if (window.api && window.api.deleteChat) {
      await window.api.deleteChat(id);
      if (currentChatId === id) {
        handleNewChat();
      }
      loadChatsFromStorage();
    }
  };

  const handleRenameChat = async (id, newTitle) => {
    if (window.api && window.api.renameChat) {
      await window.api.renameChat(id, newTitle);
      loadChatsFromStorage();
    }
  };

  const handleSendMessage = async (promptText, mode) => {
    let activeId = currentChatId;
    if (!activeId) {
      activeId = `session_${Date.now()}`;
      setCurrentChatId(activeId);
    }

    const updatedUserMessages = [...messages, { role: 'user', content: promptText }];
    setMessages(updatedUserMessages);
    setIsExecuting(true);
    setActiveTab('studio');

    if (window.api && window.api.runClaudePrompt) {
      await window.api.runClaudePrompt({
        messages: updatedUserMessages,
        model: selectedModel || 'gemma4:latest',
        workspacePath,
      });
    }
  };

  const handleCancel = async () => {
    if (window.api && window.api.cancelGrok) {
      await window.api.cancelGrok();
      setIsExecuting(false);
    }
  };

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
    }
  };

  const handleSelectFileFromTree = (file) => {
    setActiveFile(file);
    setActiveTab('studio');
    setStudioRightPane('editor');
  };

  return (
    <div className="flex h-screen w-screen bg-[#0A0C10] text-slate-100 overflow-hidden select-text font-sans">
      {/* Left Vertical Icon Rail (Matching Open Design Image 2) */}
      <IconNavRail
        activeTab={activeTab}
        onTabChange={(t) => setActiveTab(t)}
        onNewChat={handleNewChat}
      />

      {/* Main Container */}
      <div className="flex-1 flex flex-col min-w-0 h-full overflow-hidden">
        {/* Top Header Window Bar (Native macOS Dragging + Header Controls) */}
        <AppChromeHeader
          activeTab={activeTab}
          onTabChange={(t) => setActiveTab(t)}
          workspacePath={workspacePath}
          onSelectWorkspace={handleSelectWorkspace}
          selectedModel={selectedModel}
          onSelectModel={(m) => setSelectedModel(m)}
          installedModels={installedModels}
          onOpenModelManager={() => setIsModelModalOpen(true)}
        />

        {/* Content Workspace */}
        <div className="flex-1 flex overflow-hidden">
          {/* Secondary Collapsible Session Sidebar */}
          <Sidebar
            chats={chats}
            currentChatId={currentChatId}
            onNewChat={handleNewChat}
            onSelectChat={handleSelectChat}
            onDeleteChat={handleDeleteChat}
            onRenameChat={handleRenameChat}
            activeTab={activeTab}
            onTabChange={(t) => setActiveTab(t)}
          />

          {/* Core Body View */}
          <div className="flex-1 flex overflow-hidden">
            {activeTab === 'home' ? (
              /* Open Design Home Hero View (Matching Image 2) */
              <HomeView
                onSendMessage={handleSendMessage}
                workspacePath={workspacePath}
                onSelectWorkspace={handleSelectWorkspace}
              />
            ) : activeTab === 'studio' || activeTab === 'code' ? (
              /* Studio View: File Explorer + Agent Console + Right Studio Viewer (Code / Live Preview) */
              <div className="flex-1 flex overflow-hidden">
                <FileExplorer
                  workspacePath={workspacePath}
                  treeData={workspaceTree}
                  onSelectWorkspace={handleSelectWorkspace}
                  onSelectFile={handleSelectFileFromTree}
                  activeFile={activeFile}
                />

                {/* Center Chat Console */}
                <div className="flex-1 flex flex-col border-r border-white/10 min-w-[320px] overflow-hidden">
                  <AgentChat
                    messages={messages}
                    onSendMessage={handleSendMessage}
                    isExecuting={isExecuting}
                    onCancel={handleCancel}
                    selectedModel={selectedModel}
                    onSelectModel={(m) => setSelectedModel(m)}
                    installedModels={installedModels}
                    onOpenModelManager={() => setIsModelModalOpen(true)}
                    workspacePath={workspacePath}
                  />
                </div>

                {/* Right Studio Viewer (Code Editor OR Sandboxed Live Preview) */}
                <div className="w-[42%] flex flex-col bg-[#0D0E12] border-l border-white/10 min-w-[360px]">
                  <div className="h-10 border-b border-white/10 px-4 bg-black/40 flex items-center justify-between text-xs font-mono shrink-0 select-none">
                    <span className="text-slate-400 uppercase text-[11px] font-semibold">Studio Viewer</span>
                    <div className="flex bg-black/40 p-0.5 rounded-lg border border-white/10 text-xs">
                      <button
                        onClick={() => setStudioRightPane('editor')}
                        className={`px-3 py-1 rounded text-xs font-medium transition-colors ${
                          studioRightPane === 'editor' ? 'bg-white/15 text-white' : 'text-slate-400 hover:text-white'
                        }`}
                      >
                        Code Editor
                      </button>
                      <button
                        onClick={() => setStudioRightPane('preview')}
                        className={`px-3 py-1 rounded text-xs font-medium transition-colors ${
                          studioRightPane === 'preview' ? 'bg-amber-500/20 text-amber-300 border border-amber-500/30' : 'text-slate-400 hover:text-white'
                        }`}
                      >
                        Live Preview
                      </button>
                    </div>
                  </div>

                  <div className="flex-1 overflow-hidden">
                    {studioRightPane === 'editor' ? (
                      <CodeEditor
                        activeFile={activeFile}
                        onSaveFile={() => {
                          if (workspacePath) {
                            window.api.readTree(workspacePath).then((res) => {
                              if (res.success) setWorkspaceTree(res.tree);
                            });
                          }
                        }}
                      />
                    ) : (
                      <ArtifactPreview workspacePath={workspacePath} />
                    )}
                  </div>
                </div>
              </div>
            ) : activeTab === 'design-system' ? (
              /* Design System View */
              <div className="flex-1 p-8 bg-[#0A0C10] overflow-y-auto custom-scrollbar">
                <div className="max-w-3xl space-y-6">
                  <div className="flex items-center gap-3">
                    <Layers className="text-amber-400" size={28} />
                    <div>
                      <h2 className="text-2xl font-semibold text-slate-100">DESIGN.md Brand System</h2>
                      <p className="text-xs text-slate-400">Design systems and contracts that shape every generated artifact.</p>
                    </div>
                  </div>
                  <div className="p-6 rounded-2xl bg-[#131418] border border-white/10 font-mono text-xs text-slate-300">
                    {`# DESIGN.md Contract
- Primary Palette: #F59E0B (Amber), #EA580C (Orange)
- Background: #0A0C10 (Obsidian)
- Fonts: Serif display headings, Inter sans body`}
                  </div>
                </div>
              </div>
            ) : activeTab === 'automation' ? (
              /* Automation View */
              <div className="flex-1 p-8 bg-[#0A0C10] overflow-y-auto custom-scrollbar">
                <div className="max-w-3xl space-y-6">
                  <div className="flex items-center gap-3">
                    <Zap className="text-purple-400" size={28} />
                    <div>
                      <h2 className="text-2xl font-semibold text-slate-100">Automations & Routines</h2>
                      <p className="text-xs text-slate-400">Schedule automatic design builds and code tasks.</p>
                    </div>
                  </div>
                  <div className="p-6 rounded-2xl bg-[#131418] border border-white/10 text-xs text-slate-400 text-center">
                    No active background routines.
                  </div>
                </div>
              </div>
            ) : activeTab === 'plugins' ? (
              /* Plugins View */
              <div className="flex-1 p-8 bg-[#0A0C10] overflow-y-auto custom-scrollbar">
                <div className="max-w-3xl space-y-6">
                  <div className="flex items-center gap-3">
                    <Puzzle className="text-emerald-400" size={28} />
                    <div>
                      <h2 className="text-2xl font-semibold text-slate-100">Plugins & Skills</h2>
                      <p className="text-xs text-slate-400">Manage agent skills and CLI extensions.</p>
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              /* Integrations View */
              <div className="flex-1 p-8 bg-[#0A0C10] overflow-y-auto custom-scrollbar">
                <div className="max-w-3xl space-y-6">
                  <div className="flex items-center gap-3">
                    <LinkIcon className="text-blue-400" size={28} />
                    <div>
                      <h2 className="text-2xl font-semibold text-slate-100">Integrations</h2>
                      <p className="text-xs text-slate-400">Connect local Ollama servers and CLI executables.</p>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
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
