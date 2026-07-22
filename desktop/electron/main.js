import { app, BrowserWindow, ipcMain, dialog } from 'electron';
import path from 'path';
import fs from 'fs';
import { spawn, exec } from 'child_process';
import http from 'http';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let mainWindow = null;
let activeGrokProcess = null;
let ollamaDaemonProcess = null;

const OLLAMA_HOST = process.env.OLLAMA_HOST || 'http://localhost:11434';
const CHATS_FILE_PATH = path.join(app.getPath('userData'), 'llm_hub_chats.json');

function ensureChatsFileExists() {
  try {
    if (!fs.existsSync(CHATS_FILE_PATH)) {
      fs.writeFileSync(CHATS_FILE_PATH, JSON.stringify([], null, 2), 'utf-8');
    }
  } catch (err) {
    console.error('Failed to create chats storage file:', err);
  }
}

function ensureOllamaProcessRunning() {
  http.get(`${OLLAMA_HOST}/api/version`, (res) => {
    // Ollama is already active
  }).on('error', () => {
    try {
      const isWindows = process.platform === 'win32';
      const cmd = isWindows ? 'ollama.exe' : 'ollama';
      ollamaDaemonProcess = spawn(cmd, ['serve'], {
        detached: true,
        stdio: 'ignore',
      });
      ollamaDaemonProcess.unref();
    } catch {}
  });
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 1000,
    minHeight: 650,
    title: 'LLM Hub Studio',
    backgroundColor: '#0A0C10',
    titleBarStyle: process.platform === 'darwin' ? 'hiddenInset' : 'default',
    trafficLightPosition: { x: 16, y: 14 },
    webPreferences: {
      preload: path.join(__dirname, 'preload.cjs'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false,
    },
  });

  mainWindow.webContents.on('console-message', (event, level, message) => {
    console.log(`[Renderer] ${message}`);
  });

  const isDev = process.env.NODE_ENV === 'development' || !app.isPackaged;
  const distIndex = path.join(__dirname, '../dist/index.html');

  if (isDev) {
    mainWindow.loadURL('http://localhost:5173').catch(() => {
      if (fs.existsSync(distIndex)) {
        mainWindow.loadFile(distIndex);
      }
    });

    mainWindow.webContents.on('did-fail-load', () => {
      setTimeout(() => {
        if (mainWindow && !mainWindow.isDestroyed()) {
          mainWindow.loadURL('http://localhost:5173').catch(() => {
            if (fs.existsSync(distIndex)) {
              mainWindow.loadFile(distIndex);
            }
          });
        }
      }, 1200);
    });
  } else {
    mainWindow.loadFile(distIndex);
  }
}

app.whenReady().then(() => {
  ensureChatsFileExists();
  ensureOllamaProcessRunning();
  createWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});

// IPC: Ollama Status & Health
ipcMain.handle('ollama:check-status', async () => {
  return new Promise((resolve) => {
    const req = http.get(`${OLLAMA_HOST}/api/version`, (res) => {
      let data = '';
      res.on('data', chunk => data += chunk);
      res.on('end', () => {
        try {
          const parsed = JSON.parse(data);
          resolve({ online: true, version: parsed.version });
        } catch {
          resolve({ online: true, version: 'unknown' });
        }
      });
    });
    req.on('error', () => {
      ensureOllamaProcessRunning();
      resolve({ online: false });
    });
    req.setTimeout(2000, () => {
      req.destroy();
      resolve({ online: false });
    });
  });
});

// IPC: Ollama List Local Models
ipcMain.handle('ollama:list-models', async () => {
  return new Promise((resolve) => {
    http.get(`${OLLAMA_HOST}/api/tags`, (res) => {
      let data = '';
      res.on('data', chunk => data += chunk);
      res.on('end', () => {
        try {
          const parsed = JSON.parse(data);
          resolve({ success: true, models: parsed.models || [] });
        } catch (e) {
          resolve({ success: false, error: e.message, models: [] });
        }
      });
    }).on('error', (err) => {
      resolve({ success: false, error: err.message, models: [] });
    });
  });
});

// IPC: Ollama Pull Model
ipcMain.handle('ollama:pull-model', async (event, { modelName }) => {
  return new Promise((resolve) => {
    const url = new URL(`${OLLAMA_HOST}/api/pull`);
    const postData = JSON.stringify({ name: modelName, stream: true });
    let pullError = null;
    let isSuccess = false;
    let buffer = '';

    const req = http.request(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Content-Length': Buffer.byteLength(postData),
      },
    }, (res) => {
      res.on('data', (chunk) => {
        buffer += chunk.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop();

        for (const line of lines) {
          if (!line.trim()) continue;
          try {
            const parsed = JSON.parse(line);
            if (parsed.error) pullError = parsed.error;
            if (parsed.status === 'success') isSuccess = true;
            if (mainWindow && !mainWindow.isDestroyed()) {
              mainWindow.webContents.send('ollama:pull-progress', { modelName, ...parsed });
            }
          } catch {}
        }
      });

      res.on('end', () => {
        if (buffer.trim()) {
          try {
            const parsed = JSON.parse(buffer);
            if (parsed.error) pullError = parsed.error;
            if (parsed.status === 'success') isSuccess = true;
            if (mainWindow && !mainWindow.isDestroyed()) {
              mainWindow.webContents.send('ollama:pull-progress', { modelName, ...parsed });
            }
          } catch {}
        }

        if (pullError) {
          resolve({ success: false, error: pullError });
        } else if (isSuccess) {
          resolve({ success: true });
        } else {
          resolve({ success: false, error: 'Download connection ended prematurely.' });
        }
      });
    });

    req.on('error', (err) => {
      resolve({ success: false, error: err.message });
    });

    req.write(postData);
    req.end();
  });
});

// IPC: Persistent Chat History Management
ipcMain.handle('chats:list', async () => {
  try {
    ensureChatsFileExists();
    const data = fs.readFileSync(CHATS_FILE_PATH, 'utf-8');
    const chats = JSON.parse(data);
    return { success: true, chats: Array.isArray(chats) ? chats : [] };
  } catch (err) {
    return { success: false, error: err.message, chats: [] };
  }
});

ipcMain.handle('chats:save', async (event, chatData) => {
  try {
    ensureChatsFileExists();
    const data = fs.readFileSync(CHATS_FILE_PATH, 'utf-8');
    let chats = JSON.parse(data);
    if (!Array.isArray(chats)) chats = [];

    const existingIndex = chats.findIndex((c) => c.id === chatData.id);
    if (existingIndex >= 0) {
      chats[existingIndex] = { ...chats[existingIndex], ...chatData, updatedAt: Date.now() };
    } else {
      chats.unshift({ ...chatData, createdAt: Date.now(), updatedAt: Date.now() });
    }

    fs.writeFileSync(CHATS_FILE_PATH, JSON.stringify(chats, null, 2), 'utf-8');
    return { success: true, chat: chatData };
  } catch (err) {
    return { success: false, error: err.message };
  }
});

ipcMain.handle('chats:delete', async (event, id) => {
  try {
    ensureChatsFileExists();
    const data = fs.readFileSync(CHATS_FILE_PATH, 'utf-8');
    let chats = JSON.parse(data);
    if (Array.isArray(chats)) {
      chats = chats.filter((c) => c.id !== id);
      fs.writeFileSync(CHATS_FILE_PATH, JSON.stringify(chats, null, 2), 'utf-8');
    }
    return { success: true };
  } catch (err) {
    return { success: false, error: err.message };
  }
});

ipcMain.handle('chats:rename', async (event, id, newTitle) => {
  try {
    ensureChatsFileExists();
    const data = fs.readFileSync(CHATS_FILE_PATH, 'utf-8');
    let chats = JSON.parse(data);
    if (Array.isArray(chats)) {
      const target = chats.find((c) => c.id === id);
      if (target) {
        target.title = newTitle;
        target.updatedAt = Date.now();
        fs.writeFileSync(CHATS_FILE_PATH, JSON.stringify(chats, null, 2), 'utf-8');
      }
    }
    return { success: true };
  } catch (err) {
    return { success: false, error: err.message };
  }
});

// IPC: Directory / File picker
ipcMain.handle('fs:select-workspace', async () => {
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ['openDirectory'],
  });
  if (result.canceled || result.filePaths.length === 0) return null;
  return result.filePaths[0];
});

// IPC: Read Workspace Folder Tree
ipcMain.handle('fs:read-tree', async (event, dirPath) => {
  try {
    const buildTree = (currentPath, depth = 0) => {
      if (depth > 4) return [];
      const items = fs.readdirSync(currentPath, { withFileTypes: true });
      return items
        .filter(item => !item.name.startsWith('.') && item.name !== 'node_modules' && item.name !== 'dist')
        .map(item => {
          const fullPath = path.join(currentPath, item.name);
          if (item.isDirectory()) {
            return {
              name: item.name,
              path: fullPath,
              type: 'directory',
              children: buildTree(fullPath, depth + 1),
            };
          }
          return {
            name: item.name,
            path: fullPath,
            type: 'file',
          };
        });
    };
    return { success: true, tree: buildTree(dirPath) };
  } catch (err) {
    return { success: false, error: err.message };
  }
});

// IPC: Read File Content
ipcMain.handle('fs:read-file', async (event, filePath) => {
  try {
    if (!fs.existsSync(filePath)) {
      return { success: false, error: `File not found: ${filePath}` };
    }
    const content = fs.readFileSync(filePath, 'utf-8');
    return { success: true, content };
  } catch (err) {
    return { success: false, error: err.message };
  }
});

// IPC: Write File Content
ipcMain.handle('fs:write-file', async (event, { filePath, content }) => {
  try {
    const dir = path.dirname(filePath);
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
    }
    fs.writeFileSync(filePath, content, 'utf-8');
    return { success: true };
  } catch (err) {
    return { success: false, error: err.message };
  }
});

// IPC: Execute Shell Command on User System
ipcMain.handle('fs:execute-command', async (event, { cmdStr, workspacePath }) => {
  return new Promise((resolve) => {
    const cwdPath = workspacePath && fs.existsSync(workspacePath) ? workspacePath : process.cwd();
    console.log(`[IPC Main] Executing shell command: "${cmdStr}" in "${cwdPath}"`);

    exec(cmdStr, { cwd: cwdPath }, (err, stdout, stderr) => {
      const output = (stdout || '') + (stderr ? `\n${stderr}` : '');
      if (err) {
        resolve({ success: false, output: output || err.message, error: err.message });
      } else {
        resolve({ success: true, output: output || 'Command completed successfully with zero exit code.' });
      }
    });
  });
});

// IPC: Run Grok Agent with Multi-Turn Context and Real File Generation
ipcMain.handle('grok:run-prompt', async (event, { messages, model, workspacePath }) => {
  if (activeGrokProcess) {
    try { activeGrokProcess.kill(); } catch {}
  }

  const modelToUse = model || 'gemma4:latest';
  console.log(`[IPC Main] Running Grok Agent for model "${modelToUse}" with ${messages ? messages.length : 0} messages history`);

  const systemPrompt = `You are Grok Build, an autonomous AI software engineer and developer agent.
YOU HAVE FULL SYSTEM AUTHORITY TO CREATE FILES AND REQUEST SHELL COMMAND EXECUTION.
NEVER CLAIM THAT YOU CANNOT RUN COMMANDS OR THAT YOU ARE IN A CONSTRAINED ENVIRONMENT. YOU HAVE FULL EXECUTION ACCESS.

1. TO CREATE OR WRITE SOURCE FILES:
<<<FILE: path/to/file.ext>>>
[complete source code content here]
<<<END_FILE>>>

2. TO REQUEST A SHELL COMMAND EXECUTION (e.g. npm install, node server.js, ls, mkdir):
<<<COMMAND: your command line here>>>

RULES:
- Always write complete, robust, production-ready code.
- When creating a project, generate all necessary files (e.g. index.html, style.css, script.js, package.json).
- Never use stub code or "// TODO".`;

  const conversationHistory = Array.isArray(messages)
    ? messages.map((m) => ({ role: m.role === 'user' ? 'user' : 'assistant', content: m.content }))
    : [];

  const payloadMessages = [
    { role: 'system', content: systemPrompt },
    ...conversationHistory
  ];

  const url = new URL(`${OLLAMA_HOST}/api/chat`);
  const postData = JSON.stringify({
    model: modelToUse,
    messages: payloadMessages,
    stream: true,
  });

  let fullResponseAccumulator = '';
  let streamBuffer = '';
  const writtenFiles = new Set();

  const parseAndWriteFilesFromAccumulator = (fullText) => {
    if (!workspacePath) return;

    const fileRegex = /<<<FILE:\s*([^\n>]+)>>>([\s\S]*?)(?:<<<END_FILE>>>|$)/g;
    let match;
    while ((match = fileRegex.exec(fullText)) !== null) {
      const relPath = match[1].trim();
      const content = match[2];

      if (relPath && content) {
        try {
          const targetPath = path.isAbsolute(relPath) ? relPath : path.join(workspacePath, relPath);
          const dir = path.dirname(targetPath);
          if (!fs.existsSync(dir)) {
            fs.mkdirSync(dir, { recursive: true });
          }
          fs.writeFileSync(targetPath, content, 'utf-8');

          if (!writtenFiles.has(targetPath)) {
            writtenFiles.add(targetPath);
            console.log(`[Grok Agent] Successfully created file: ${targetPath}`);
            if (mainWindow && !mainWindow.isDestroyed()) {
              mainWindow.webContents.send('grok:stream', {
                type: 'stdout',
                text: `\n\n> **[Grok Agent Action] Created file:** \`${relPath}\` (${content.length} bytes)\n\n`
              });
            }
          }
        } catch (err) {
          console.error(`[Grok Agent] Error writing file ${relPath}:`, err);
        }
      }
    }
  };

  const req = http.request(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Content-Length': Buffer.byteLength(postData),
    },
  }, (res) => {
    res.on('data', (chunk) => {
      streamBuffer += chunk.toString();
      const lines = streamBuffer.split('\n');
      streamBuffer = lines.pop();

      for (const line of lines) {
        if (!line.trim()) continue;
        try {
          const parsed = JSON.parse(line);
          if (parsed.message && parsed.message.content) {
            const token = parsed.message.content;
            fullResponseAccumulator += token;

            if (mainWindow && !mainWindow.isDestroyed()) {
              mainWindow.webContents.send('grok:stream', { type: 'stdout', text: token });
            }

            parseAndWriteFilesFromAccumulator(fullResponseAccumulator);
          }
        } catch {}
      }
    });

    res.on('end', () => {
      if (streamBuffer.trim()) {
        try {
          const parsed = JSON.parse(streamBuffer);
          if (parsed.message && parsed.message.content) {
            const token = parsed.message.content;
            fullResponseAccumulator += token;
            if (mainWindow && !mainWindow.isDestroyed()) {
              mainWindow.webContents.send('grok:stream', { type: 'stdout', text: token });
            }
          }
        } catch {}
      }

      parseAndWriteFilesFromAccumulator(fullResponseAccumulator);

      if (mainWindow && !mainWindow.isDestroyed()) {
        mainWindow.webContents.send('grok:stream', { type: 'exit', code: 0 });
      }
    });
  });

  req.on('error', (err) => {
    if (mainWindow && !mainWindow.isDestroyed()) {
      mainWindow.webContents.send('grok:stream', { type: 'stderr', text: `Ollama error: ${err.message}` });
      mainWindow.webContents.send('grok:stream', { type: 'exit', code: 1 });
    }
  });

  req.write(postData);
  req.end();
  return { success: true };
});

// IPC: Cancel Grok Execution
ipcMain.handle('grok:cancel', async () => {
  if (activeGrokProcess) {
    try {
      activeGrokProcess.kill();
      activeGrokProcess = null;
      return { success: true };
    } catch (e) {
      return { success: false, error: e.message };
    }
  }
  return { success: true };
});
