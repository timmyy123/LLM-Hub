import React, { useState, useEffect } from 'react';
import Sidebar from './components/Sidebar';
import FileExplorer from './components/FileExplorer';
import AgentChat from './components/AgentChat';
import CodeEditor from './components/CodeEditor';
import ModelManagerModal from './components/ModelManagerModal';
import { filterAllowedModels } from './services/ollamaService';

export default function App() {
  const [ollamaOnline, setOllamaOnline] = useState(false);
  const [installedModels, setInstalledModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState(null);
  const [workspacePath, setWorkspacePath] = useState('');
  const [workspaceTree, setWorkspaceTree] = useState([]);
  const [activeFile, setActiveFile] = useState(null);
  const [isModelModalOpen, setIsModelModalOpen] = useState(false);

  // Chat History & Active Chat State
  const [chats, setChats] = useState([]);
  const [currentChatId, setCurrentChatId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [isExecuting, setIsExecuting] = useState(false);
  const [activeSidebarTab, setActiveSidebarTab] = useState('home'); // 'home' | 'code'

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
    const title = currentMessages[0]?.content?.slice(0, 32) || 'New Chat';
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
    const newId = `chat_${Date.now()}`;
    setCurrentChatId(newId);
    setMessages([]);
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
      activeId = `chat_${Date.now()}`;
      setCurrentChatId(activeId);
    }

    const updatedUserMessages = [...messages, { role: 'user', content: promptText }];
    setMessages(updatedUserMessages);
    setIsExecuting(true);

    if (window.api && window.api.runGrokPrompt) {
      await window.api.runGrokPrompt({
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
    setActiveSidebarTab('code');
  };

  return (
    <div className="flex h-screen w-screen bg-[#0A0C10] text-slate-100 overflow-hidden select-text">
      {/* Sidebar */}
      <Sidebar
        chats={chats}
        currentChatId={currentChatId}
        onNewChat={handleNewChat}
        onSelectChat={handleSelectChat}
        onDeleteChat={handleDeleteChat}
        onRenameChat={handleRenameChat}
        activeTab={activeSidebarTab}
        onTabChange={(t) => setActiveSidebarTab(t)}
      />

      {/* Main Content Workspace */}
      <div className="flex-1 flex overflow-hidden">
        {activeSidebarTab === 'home' ? (
          /* Home View: Grok Agent Chat Workspace */
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
        ) : (
          /* Code View: File Explorer + Code Editor */
          <div className="flex-1 flex overflow-hidden">
            <FileExplorer
              workspacePath={workspacePath}
              treeData={workspaceTree}
              onSelectWorkspace={handleSelectWorkspace}
              onSelectFile={handleSelectFileFromTree}
              activeFile={activeFile}
            />

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
          </div>
        )}
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
