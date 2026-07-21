import { app, BrowserWindow, ipcMain, dialog } from 'electron';
import path from 'path';
import fs from 'fs';
import { spawn } from 'child_process';
import http from 'http';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let mainWindow = null;
let activeGrokProcess = null;
let ollamaDaemonProcess = null;

const OLLAMA_HOST = process.env.OLLAMA_HOST || 'http://localhost:11434';

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
          console.log(`[IPC Main] ollama:list-models returning ${parsed.models ? parsed.models.length : 0} models`);
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

// IPC: Ollama Pull Model (with line buffering & error handling)
ipcMain.handle('ollama:pull-model', async (event, { modelName }) => {
  console.log(`[IPC Main] ollama:pull-model received request for "${modelName}"`);
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
      console.log(`[IPC Main] Ollama pull connection established for "${modelName}". HTTP Status: ${res.statusCode}`);

      res.on('data', (chunk) => {
        buffer += chunk.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop();

        for (const line of lines) {
          if (!line.trim()) continue;
          try {
            const parsed = JSON.parse(line);
            if (parsed.error) {
              pullError = parsed.error;
              console.error(`[IPC Main] Ollama returned error line for "${modelName}":`, parsed.error);
            }
            if (parsed.status === 'success') {
              isSuccess = true;
              console.log(`[IPC Main] Ollama returned 'success' status line for "${modelName}"`);
            }
            if (mainWindow && !mainWindow.isDestroyed()) {
              mainWindow.webContents.send('ollama:pull-progress', { modelName, ...parsed });
            }
          } catch (e) {
            console.error(`[IPC Main] Failed to parse JSON stream line:`, line);
          }
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

        console.log(`[IPC Main] Ollama pull stream ended for "${modelName}". isSuccess: ${isSuccess}, pullError: ${pullError}`);
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
      console.error(`[IPC Main] HTTP request error pulling "${modelName}":`, err.message);
      resolve({ success: false, error: err.message });
    });

    req.write(postData);
    req.end();
  });
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
    const content = fs.readFileSync(filePath, 'utf-8');
    return { success: true, content };
  } catch (err) {
    return { success: false, error: err.message };
  }
});

// IPC: Run Agent Execution with Ollama direct streaming fallback
ipcMain.handle('grok:run-prompt', async (event, { prompt, model, workspacePath }) => {
  if (activeGrokProcess) {
    try { activeGrokProcess.kill(); } catch {}
  }

  const modelToUse = model || 'gemma4:latest';
  console.log(`[IPC Main] grok:run-prompt called for model "${modelToUse}"`);

  const runOllamaDirectStream = () => {
    console.log(`[IPC Main] Direct Ollama stream executing for model "${modelToUse}"...`);
    const url = new URL(`${OLLAMA_HOST}/api/chat`);
    const postData = JSON.stringify({
      model: modelToUse,
      messages: [{ role: 'user', content: prompt }],
      stream: true,
    });

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
            if (parsed.message && parsed.message.content) {
              if (mainWindow && !mainWindow.isDestroyed()) {
                mainWindow.webContents.send('grok:stream', { type: 'stdout', text: parsed.message.content });
              }
            }
          } catch {}
        }
      });

      res.on('end', () => {
        if (buffer.trim()) {
          try {
            const parsed = JSON.parse(buffer);
            if (parsed.message && parsed.message.content) {
              if (mainWindow && !mainWindow.isDestroyed()) {
                mainWindow.webContents.send('grok:stream', { type: 'stdout', text: parsed.message.content });
              }
            }
          } catch {}
        }
        if (mainWindow && !mainWindow.isDestroyed()) {
          mainWindow.webContents.send('grok:stream', { type: 'exit', code: 0 });
        }
      });
    });

    req.on('error', (err) => {
      if (mainWindow && !mainWindow.isDestroyed()) {
        mainWindow.webContents.send('grok:stream', { type: 'stderr', text: `Ollama connection error: ${err.message}` });
        mainWindow.webContents.send('grok:stream', { type: 'exit', code: 1 });
      }
    });

    req.write(postData);
    req.end();
    return { success: true, mode: 'ollama-direct' };
  };

  const isWindows = process.platform === 'win32';
  const command = isWindows ? 'grok-build.cmd' : 'grok-build';

  try {
    const env = {
      ...process.env,
      OPENAI_BASE_URL: `${OLLAMA_HOST}/v1`,
      OPENAI_API_KEY: 'ollama',
      OLLAMA_HOST: OLLAMA_HOST,
      GROK_MODEL: modelToUse,
    };

    let cliSpawned = false;
    let proc = spawn(command, ['run', '--model', modelToUse, '--prompt', prompt], {
      cwd: workspacePath || process.cwd(),
      env,
      shell: true,
    });
    activeGrokProcess = proc;

    proc.on('error', () => {
      if (!cliSpawned) {
        runOllamaDirectStream();
      }
    });

    proc.stdout.on('data', (data) => {
      cliSpawned = true;
      if (mainWindow && !mainWindow.isDestroyed()) {
        mainWindow.webContents.send('grok:stream', { type: 'stdout', text: data.toString() });
      }
    });

    proc.stderr.on('data', (data) => {
      const text = data.toString();
      if (text.includes('command not found') || text.includes('not recognized')) {
        if (!cliSpawned) {
          try { proc.kill(); } catch {}
          runOllamaDirectStream();
          return;
        }
      }
      cliSpawned = true;
      if (mainWindow && !mainWindow.isDestroyed()) {
        mainWindow.webContents.send('grok:stream', { type: 'stderr', text });
      }
    });

    proc.on('close', (code) => {
      activeGrokProcess = null;
      if (cliSpawned) {
        if (mainWindow && !mainWindow.isDestroyed()) {
          mainWindow.webContents.send('grok:stream', { type: 'exit', code });
        }
      }
    });

    return { success: true, pid: proc.pid };
  } catch (err) {
    return runOllamaDirectStream();
  }
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
