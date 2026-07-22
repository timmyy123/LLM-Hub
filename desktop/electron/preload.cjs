const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('api', {
  checkOllamaStatus: () => ipcRenderer.invoke('ollama:check-status'),
  listModels: () => ipcRenderer.invoke('ollama:list-models'),
  pullModel: (modelName) => ipcRenderer.invoke('ollama:pull-model', { modelName }),
  onPullProgress: (callback) => {
    const listener = (_event, value) => callback(value);
    ipcRenderer.on('ollama:pull-progress', listener);
    return () => ipcRenderer.removeListener('ollama:pull-progress', listener);
  },

  selectWorkspace: () => ipcRenderer.invoke('fs:select-workspace'),
  readTree: (dirPath) => ipcRenderer.invoke('fs:read-tree', dirPath),
  readFile: (filePath) => ipcRenderer.invoke('fs:read-file', filePath),
  writeFile: (filePath, content) => ipcRenderer.invoke('fs:write-file', { filePath, content }),
  executeCommand: (cmdStr, workspacePath) => ipcRenderer.invoke('fs:execute-command', { cmdStr, workspacePath }),

  // Persistent Chat History IPCs
  listChats: () => ipcRenderer.invoke('chats:list'),
  saveChat: (chatData) => ipcRenderer.invoke('chats:save', chatData),
  deleteChat: (id) => ipcRenderer.invoke('chats:delete', id),
  renameChat: (id, title) => ipcRenderer.invoke('chats:rename', id, title),

  // Agent Execution IPCs
  runGrokPrompt: ({ messages, model, workspacePath }) =>
    ipcRenderer.invoke('grok:run-prompt', { messages, model, workspacePath }),
  cancelGrok: () => ipcRenderer.invoke('grok:cancel'),
  onGrokStream: (callback) => {
    const listener = (_event, value) => callback(value);
    ipcRenderer.on('grok:stream', listener);
    return () => ipcRenderer.removeListener('grok:stream', listener);
  },
});
