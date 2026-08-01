import React, { useState, useEffect } from 'react';
import { RefreshCw, ExternalLink, Globe, Monitor, Smartphone, Tablet } from 'lucide-react';

export default function ArtifactPreview({ workspacePath }) {
  const [htmlContent, setHtmlContent] = useState('');
  const [deviceMode, setDeviceMode] = useState('desktop'); // 'desktop' | 'tablet' | 'mobile'
  const [keyIndex, setKeyIndex] = useState(0);

  const loadWorkspaceHtml = async () => {
    if (!workspacePath) return;
    if (window.api && window.api.readFile) {
      // Look for index.html or main html file
      const res = await window.api.readFile(`${workspacePath}/index.html`);
      if (res.success && typeof res.content === 'string') {
        // Inject base tag or inline scripts/css if needed
        setHtmlContent(res.content);
      } else {
        setHtmlContent(`<!DOCTYPE html>
<html>
  <head>
    <style>
      body { background: #0A0C10; color: #64748B; font-family: system-ui, sans-serif; display: flex; align-items: center; justify-content: center; height: 100vh; margin: 0; }
    </style>
  </head>
  <body>
    <div style="text-align:center;">
      <h2>No Live Prototype HTML Found</h2>
      <p>Ask Claude Code Agent to build a website or UI artifact to preview it live here.</p>
    </div>
  </body>
</html>`);
      }
    }
  };

  useEffect(() => {
    loadWorkspaceHtml();
  }, [workspacePath, keyIndex]);

  const handleRefresh = () => {
    setKeyIndex((prev) => prev + 1);
  };

  const getContainerWidth = () => {
    switch (deviceMode) {
      case 'mobile': return 'w-[375px] h-[667px]';
      case 'tablet': return 'w-[768px] h-[900px]';
      default: return 'w-full h-full';
    }
  };

  return (
    <div className="flex-1 flex flex-col h-full bg-[#0D0E12] text-slate-200 overflow-hidden select-none">
      {/* Chrome Top Bar */}
      <div className="h-10 border-b border-white/10 px-4 bg-black/40 flex items-center justify-between text-xs">
        <div className="flex items-center gap-2">
          <Globe size={14} className="text-emerald-400" />
          <span className="font-semibold text-slate-200 font-mono">Live Sandbox Artifact Preview</span>
        </div>

        <div className="flex items-center gap-2">
          {/* Device Frame Switcher */}
          <div className="flex bg-black/40 p-0.5 rounded-lg border border-white/10 text-xs">
            <button
              onClick={() => setDeviceMode('desktop')}
              className={`p-1 rounded ${deviceMode === 'desktop' ? 'bg-white/15 text-white' : 'text-slate-400 hover:text-white'}`}
              title="Desktop View"
            >
              <Monitor size={13} />
            </button>
            <button
              onClick={() => setDeviceMode('tablet')}
              className={`p-1 rounded ${deviceMode === 'tablet' ? 'bg-white/15 text-white' : 'text-slate-400 hover:text-white'}`}
              title="Tablet View"
            >
              <Tablet size={13} />
            </button>
            <button
              onClick={() => setDeviceMode('mobile')}
              className={`p-1 rounded ${deviceMode === 'mobile' ? 'bg-white/15 text-white' : 'text-slate-400 hover:text-white'}`}
              title="Mobile View"
            >
              <Smartphone size={13} />
            </button>
          </div>

          <button
            onClick={handleRefresh}
            className="p-1.5 rounded-lg bg-white/10 hover:bg-white/20 text-slate-300 transition-colors"
            title="Refresh Preview"
          >
            <RefreshCw size={13} />
          </button>
        </div>
      </div>

      {/* Sandboxed Live Iframe */}
      <div className="flex-1 flex items-center justify-center p-4 bg-[#0A0C10] overflow-auto custom-scrollbar">
        <div className={`transition-all duration-300 shadow-2xl rounded-xl overflow-hidden border border-white/10 bg-white ${getContainerWidth()}`}>
          <iframe
            key={keyIndex}
            srcDoc={htmlContent}
            title="Claude Code Live Artifact"
            className="w-full h-full border-none"
            sandbox="allow-scripts allow-modals allow-same-origin"
          />
        </div>
      </div>
    </div>
  );
}
