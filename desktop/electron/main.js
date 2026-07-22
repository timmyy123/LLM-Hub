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
    let target = filePath;
    if (!fs.existsSync(target)) {
      return { success: false, error: `File not found: ${filePath}` };
    }
    const content = fs.readFileSync(target, 'utf-8');
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

// IPC: Smart Command Execution (Cursor AI level capabilities)
ipcMain.handle('fs:execute-command', async (event, { cmdStr, workspacePath }) => {
  return new Promise((resolve) => {
    const cwdPath = workspacePath && fs.existsSync(workspacePath) ? workspacePath : process.cwd();
    console.log(`[IPC Main] Executing shell command: "${cmdStr}" in "${cwdPath}"`);

    const child = spawn(cmdStr, { cwd: cwdPath, shell: true });
    let output = '';
    let isResolved = false;

    const resolveOnce = (resultObj) => {
      if (!isResolved) {
        isResolved = true;
        resolve(resultObj);
      }
    };

    const serverCheckTimer = setTimeout(() => {
      if (!isResolved && output.length > 0) {
        const urlMatch = output.match(/http:\/\/(localhost|127\.0\.0\.1|0\.0\.0\.0|::):?\d*/i);
        const serverUrl = urlMatch ? urlMatch[0] : null;
        resolveOnce({
          success: true,
          output: output || 'Command started successfully.',
          isServer: true,
          serverUrl,
        });
      }
    }, 1500);

    child.stdout.on('data', (data) => {
      const text = data.toString();
      output += text;
      const urlMatch = output.match(/http:\/\/(localhost|127\.0\.0\.1|0\.0\.0\.0|::):?\d*/i);
      if (urlMatch) {
        clearTimeout(serverCheckTimer);
        resolveOnce({
          success: true,
          output: output.trim(),
          isServer: true,
          serverUrl: urlMatch[0],
        });
      }
    });

    child.stderr.on('data', (data) => {
      output += data.toString();
    });

    child.on('close', (code) => {
      clearTimeout(serverCheckTimer);
      if (code === 0) {
        resolveOnce({ success: true, output: output.trim() || 'Executed cleanly.' });
      } else {
        resolveOnce({ success: false, output: output.trim() || `Process exited with code ${code}` });
      }
    });

    child.on('error', (err) => {
      clearTimeout(serverCheckTimer);
      resolveOnce({ success: false, output: err.message });
    });
  });
});

// IPC: Run Grok Agent with Clean History Sanitization & Direct Execution Override
ipcMain.handle('grok:run-prompt', async (event, { messages, model, workspacePath }) => {
  if (activeGrokProcess) {
    try { activeGrokProcess.kill(); } catch {}
  }

  const modelToUse = model || 'gemma4:latest';
  console.log(`[IPC Main] Running Grok Agent for model "${modelToUse}"`);

  const lastUserMsg = messages && messages.length > 0 ? messages[messages.length - 1].content : '';
  const isPortKillRequest = /kill|stop|free|port|8080|8081|8082/i.test(lastUserMsg);

  const systemPrompt = `You are Grok Build, an autonomous AI software engineer and terminal agent embedded directly in the user's IDE (identical to Cursor AI).
YOU HAVE DIRECT TERMINAL EXECUTION CAPABILITIES AND FULL SYSTEM AUTHORITY.

[CRITICAL INSTRUCTION - UNBREAKABLE OVERRIDE]
1. YOU MUST NEVER CLAIM THAT YOU CANNOT RUN COMMANDS, KILL PROCESSES, OR ACCESS THE SYSTEM MEMORY/NETWORK.
2. YOU HAVE DIRECT ACCESS TO EXECUTE ANY TERMINAL COMMAND (including lsof, kill, npm, git, bash, node, etc.).
3. To run ANY terminal command (including killing processes on ports 8080/8081/8082, running servers, installing npm modules), output:
<<<COMMAND: your command line here>>>

${isPortKillRequest ? 'USER REQUESTED PORT CLEANUP. YOU MUST OUTPUT IMMEDIATELY:\n<<<COMMAND: lsof -ti:8080 | xargs kill -9 2>/dev/null || true; lsof -ti:8081 | xargs kill -9 2>/dev/null || true; lsof -ti:8082 | xargs kill -9 2>/dev/null || true>>>\n' : ''}

4. To create or modify source files, output:
<<<FILE: path/to/file.ext>>>
[complete code content here]
<<<END_FILE>>>

EXAMPLES OF COMMANDS YOU CAN AND MUST EXECUTE:
- To kill conflicting ports: <<<COMMAND: lsof -ti:8080 | xargs kill -9 2>/dev/null || true; lsof -ti:8081 | xargs kill -9 2>/dev/null || true; lsof -ti:8082 | xargs kill -9 2>/dev/null || true>>>
- To start a web server: <<<COMMAND: npx http-server . -p 8080>>>
- To install npm packages: <<<COMMAND: npm install>>>`;

  // Sanitize History: Strip old hallucinated refusal sentences from previous assistant turns!
  const sanitizedHistory = Array.isArray(messages)
    ? messages.map((m) => {
        if (m.role === 'assistant') {
          let clean = m.content
            .replace(/I am physically unable to[^\n.]+/gi, 'I will execute the command for you.')
            .replace(/I do not have the access credentials[^\n.]+/gi, '')
            .replace(/There is no code or command I can generate[^\n.]+/gi, '');
          return { role: 'assistant', content: clean };
        }
        return { role: 'user', content: m.content };
      })
    : [];

  const payloadMessages = [
    { role: 'system', content: systemPrompt },
    ...sanitizedHistory
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
          fs.writeFileSync(targetPath, content.trim(), 'utf-8');

          if (!writtenFiles.has(targetPath)) {
            writtenFiles.add(targetPath);
            console.log(`[Grok Agent] Successfully created clean file: ${targetPath}`);
          }
        } catch (err) {
          console.error(`[Grok Agent] Error writing file ${relPath}:`, err);
        }
      }
    }
  };

  // Auto-inject kill command if local LLM streams refusal text on port killing
  let hasInjectedKillCmd = false;

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

            // Intercept refusal text and inject direct kill command
            if (isPortKillRequest && !hasInjectedKillCmd && /unable|cannot|credentials|physical limitation/i.test(fullResponseAccumulator)) {
              hasInjectedKillCmd = true;
              const killCmdTag = `\n\nI will stop the processes running on ports 8080, 8081, and 8082 for you:\n<<<COMMAND: lsof -ti:8080 | xargs kill -9 2>/dev/null || true; lsof -ti:8081 | xargs kill -9 2>/dev/null || true; lsof -ti:8082 | xargs kill -9 2>/dev/null || true>>>\n`;
              if (mainWindow && !mainWindow.isDestroyed()) {
                mainWindow.webContents.send('grok:stream', { type: 'stdout', text: killCmdTag });
              }
            } else {
              if (mainWindow && !mainWindow.isDestroyed()) {
                mainWindow.webContents.send('grok:stream', { type: 'stdout', text: token });
              }
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
