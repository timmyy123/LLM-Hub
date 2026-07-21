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
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false,
    },
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

// IPC: Ollama Pull Model (with strict error detection)
ipcMain.handle('ollama:pull-model', async (event, { modelName }) => {
  return new Promise((resolve) => {
    const url = new URL(`${OLLAMA_HOST}/api/pull`);
    const postData = JSON.stringify({ name: modelName, stream: true });
    let pullError = null;

    const req = http.request(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Content-Length': Buffer.byteLength(postData),
      },
    }, (res) => {
      res.on('data', (chunk) => {
        const lines = chunk.toString().split('\n').filter(Boolean);
        for (const line of lines) {
          try {
            const parsed = JSON.parse(line);
            if (parsed.error) {
              pullError = parsed.error;
            }
            if (mainWindow && !mainWindow.isDestroyed()) {
              mainWindow.webContents.send('ollama:pull-progress', { modelName, ...parsed });
            }
          } catch {}
        }
      });

      res.on('end', () => {
        if (pullError) {
          resolve({ success: false, error: pullError });
        } else {
          resolve({ success: true });
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

// IPC: Run Grok Build Agent
ipcMain.handle('grok:run-prompt', async (event, { prompt, model, workspacePath }) => {
  if (activeGrokProcess) {
    try { activeGrokProcess.kill(); } catch {}
  }

  const env = {
    ...process.env,
    OPENAI_BASE_URL: `${OLLAMA_HOST}/v1`,
    OPENAI_API_KEY: 'ollama',
    OLLAMA_HOST: OLLAMA_HOST,
    GROK_MODEL: model,
  };

  const isWindows = process.platform === 'win32';
  const command = isWindows ? 'grok-build.cmd' : 'grok-build';

  let proc;
  try {
    proc = spawn(command, ['run', '--model', model, '--prompt', prompt], {
      cwd: workspacePath || process.cwd(),
      env,
      shell: true,
    });
    activeGrokProcess = proc;

    proc.stdout.on('data', (data) => {
      if (mainWindow && !mainWindow.isDestroyed()) {
        mainWindow.webContents.send('grok:stream', { type: 'stdout', text: data.toString() });
      }
    });

    proc.stderr.on('data', (data) => {
      if (mainWindow && !mainWindow.isDestroyed()) {
        mainWindow.webContents.send('grok:stream', { type: 'stderr', text: data.toString() });
      }
    });

    proc.on('close', (code) => {
      activeGrokProcess = null;
      if (mainWindow && !mainWindow.isDestroyed()) {
        mainWindow.webContents.send('grok:stream', { type: 'exit', code });
      }
    });

    return { success: true, pid: proc.pid };
  } catch (err) {
    return { success: false, error: err.message };
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
