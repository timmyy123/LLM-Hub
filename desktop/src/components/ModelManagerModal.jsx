import React, { useState } from 'react';
import { X, Download, RefreshCw, Cpu, HardDrive } from 'lucide-react';
import { ALLOWED_FAMILIES } from '../services/ollamaService';

export default function ModelManagerModal({ isOpen, onClose, installedModels, selectedModel, onSelectModel, onRefresh }) {
  const [downloadingTag, setDownloadingTag] = useState(null);
  const [downloadProgress, setDownloadProgress] = useState(null);
  const [customTagInput, setCustomTagInput] = useState('');
  const [errorMessage, setErrorMessage] = useState('');

  if (!isOpen) return null;

  const isInstalled = (tag) => {
    if (!Array.isArray(installedModels) || installedModels.length === 0) return false;
    const target = tag.toLowerCase().replace(/^library\//, '');

    const found = installedModels.some((m) => {
      const rawName = (typeof m === 'string' ? m : (m.name || m.model || '')).toLowerCase();
      const name = rawName.replace(/^library\//, '');

      // Exact tag match (e.g. "gemma4:e2b" === "gemma4:e2b")
      if (name === target) return true;

      // Base tag equivalency (gemma4 === gemma4:latest)
      if (
        (target === 'gemma4:latest' || target === 'gemma4') &&
        (name === 'gemma4' || name === 'gemma4:latest')
      ) return true;

      if (
        (target === 'ministral-3:latest' || target === 'ministral-3') &&
        (name === 'ministral-3' || name === 'ministral-3:latest')
      ) return true;

      if (
        (target === 'lfm2:latest' || target === 'lfm2') &&
        (name === 'lfm2' || name === 'lfm2:latest')
      ) return true;

      return false;
    });

    console.log(`[Modal] isInstalled tag "${tag}":`, found);
    return found;
  };

  const handlePullModel = async (tagToPull) => {
    console.log(`[Modal] Starting pull for model tag: "${tagToPull}"`);
    setErrorMessage('');
    setDownloadingTag(tagToPull);
    setDownloadProgress({ status: 'Connecting to Ollama...', percent: 0 });
    let streamError = null;
    let maxTotalBytes = 0;

    const processProgressData = (data) => {
      console.log(`[Modal] Stream Progress Event for "${tagToPull}":`, data);
      if (data.error) {
        streamError = data.error;
        console.error(`[Modal] Stream Error for "${tagToPull}":`, data.error);
        setErrorMessage(`Ollama Error: ${data.error}`);
        return;
      }

      if (data.status === 'success') {
        console.log(`[Modal] Received 'success' status from Ollama for "${tagToPull}"`);
        setDownloadProgress({ status: 'Download Complete', percent: 100 });
        return;
      }

      if (data.total && data.completed) {
        if (data.total > maxTotalBytes) {
          maxTotalBytes = data.total;
        }

        if (data.total >= 5000000 || data.total === maxTotalBytes) {
          const rawPct = Math.round((data.completed / data.total) * 100);
          const pct = Math.min(rawPct, 99);
          const currentStatus = data.status || `Downloading (${pct}%)`;
          setDownloadProgress({ status: currentStatus, percent: pct });
        } else {
          setDownloadProgress((prev) => ({
            status: data.status || prev?.status || 'Downloading...',
            percent: prev?.percent || 0,
          }));
        }
      } else if (data.status) {
        setDownloadProgress((prev) => ({
          status: data.status,
          percent: prev?.percent || 0,
        }));
      }
    };

    if (window.api && window.api.pullModel) {
      const unsub = window.api.onPullProgress((data) => {
        processProgressData(data);
      });

      console.log(`[Modal] Invoking IPC pullModel for "${tagToPull}"...`);
      const res = await window.api.pullModel(tagToPull);
      console.log(`[Modal] IPC pullModel finished with result:`, res);
      unsub();

      if (res.success && !streamError) {
        console.log(`[Modal] Download succeeded for "${tagToPull}". Refreshing models list in 800ms...`);
        setDownloadProgress({ status: 'Download Complete', percent: 100 });
        setTimeout(() => {
          setDownloadingTag(null);
          setDownloadProgress(null);
          onRefresh();
        }, 800);
      } else {
        const finalErr = streamError || res.error || `Failed to download '${tagToPull}'.`;
        console.error(`[Modal] Download failed for "${tagToPull}":`, finalErr);
        setErrorMessage(finalErr);
        setDownloadingTag(null);
        setDownloadProgress(null);
      }
    } else {
      try {
        console.log(`[Modal] Fallback direct fetch to http://localhost:11434/api/pull for "${tagToPull}"...`);
        const response = await fetch('http://localhost:11434/api/pull', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ name: tagToPull, stream: true }),
        });

        if (!response.ok) throw new Error(`HTTP Error ${response.status}`);

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n');
          buffer = lines.pop();

          for (const line of lines) {
            if (!line.trim()) continue;
            try {
              const parsed = JSON.parse(line);
              processProgressData(parsed);
            } catch {}
          }
        }

        if (!streamError) {
          console.log(`[Modal] Direct fetch pull complete for "${tagToPull}". Refreshing models...`);
          setDownloadProgress({ status: 'Download Complete', percent: 100 });
          setTimeout(() => {
            setDownloadingTag(null);
            setDownloadProgress(null);
            onRefresh();
          }, 800);
        } else {
          setDownloadingTag(null);
          setDownloadProgress(null);
        }
      } catch (err) {
        console.error(`[Modal] Direct fetch error for "${tagToPull}":`, err.message);
        setErrorMessage(`Ollama service error: ${err.message}`);
        setDownloadingTag(null);
        setDownloadProgress(null);
      }
    }
  };

  const handleCustomPull = (e) => {
    e.preventDefault();
    if (!customTagInput.trim()) return;
    handlePullModel(customTagInput.trim());
    setCustomTagInput('');
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-xl p-4">
      <div className="liquid-glass-card rounded-2xl w-full max-w-3xl overflow-hidden flex flex-col max-h-[85vh] text-slate-100">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-white/10 bg-white/5">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-white/10 flex items-center justify-center text-slate-200">
              <Cpu size={18} />
            </div>
            <div>
              <h2 className="text-sm font-semibold tracking-tight">Ollama Local Models</h2>
              <p className="text-xs text-slate-400 font-sans">Manage and download local models</p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={onRefresh}
              className="p-1.5 rounded-lg text-slate-400 hover:text-slate-200 hover:bg-white/10 transition-colors"
              title="Refresh Models"
            >
              <RefreshCw size={15} />
            </button>
            <button
              onClick={onClose}
              className="p-1.5 rounded-lg text-slate-400 hover:text-slate-200 hover:bg-white/10 transition-colors"
            >
              <X size={18} />
            </button>
          </div>
        </div>

        {/* Error Notification */}
        {errorMessage && (
          <div className="mx-6 mt-4 p-3 rounded-xl bg-rose-500/10 border border-rose-500/20 text-rose-300 text-xs font-mono">
            {errorMessage}
          </div>
        )}

        {/* Real Download Progress Bar */}
        {downloadingTag && downloadProgress && (
          <div className="mx-6 mt-4 p-4 rounded-xl bg-white/5 border border-white/10 space-y-2 font-mono">
            <div className="flex justify-between text-xs text-slate-300">
              <span>Pulling {downloadingTag} from Ollama...</span>
              <span>{downloadProgress.percent}%</span>
            </div>
            <div className="w-full bg-black/40 h-1.5 rounded-full overflow-hidden">
              <div
                className="bg-slate-200 h-full transition-all duration-300 rounded-full"
                style={{ width: `${downloadProgress.percent}%` }}
              />
            </div>
            <p className="text-[11px] text-slate-400">{downloadProgress.status}</p>
          </div>
        )}

        {/* Model Families Body */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {Object.values(ALLOWED_FAMILIES).map((family) => (
            <div key={family.id} className="bg-white/5 border border-white/10 rounded-xl p-4 space-y-3">
              <div>
                <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300 flex items-center gap-2">
                  <HardDrive size={14} />
                  {family.name}
                </h3>
                <p className="text-xs text-slate-400 mt-0.5 font-sans">{family.description}</p>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 pt-1">
                {family.popularTags.map((tag) => {
                  const installed = isInstalled(tag);
                  const isSelected = selectedModel === tag;
                  const isPullingThis = downloadingTag === tag;

                  return (
                    <div
                      key={tag}
                      className={`flex items-center justify-between p-2.5 rounded-xl border transition-all ${
                        installed
                          ? 'bg-white/10 border-white/20 text-white'
                          : 'bg-black/30 border-white/5 text-slate-300 hover:border-white/15'
                      }`}
                    >
                      <div className="flex items-center gap-2 min-w-0">
                        <span className="font-mono text-xs truncate">{tag}</span>
                        {installed ? (
                          <span className="text-[10px] px-1.5 py-0.5 rounded bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 font-medium">
                            Installed
                          </span>
                        ) : (
                          <span className="text-[10px] px-1.5 py-0.5 rounded bg-slate-800/80 text-slate-400 border border-slate-700/60">
                            Not Installed
                          </span>
                        )}
                      </div>

                      <div className="flex items-center gap-2 shrink-0">
                        {installed ? (
                          <button
                            onClick={() => onSelectModel(tag)}
                            className={`px-3 py-1 text-xs rounded-lg font-medium transition-all ${
                              isSelected
                                ? 'bg-white text-slate-950 font-semibold'
                                : 'bg-white/10 text-slate-200 hover:bg-white/20'
                            }`}
                          >
                            {isSelected ? 'Active' : 'Select'}
                          </button>
                        ) : (
                          <button
                            disabled={isPullingThis || !!downloadingTag}
                            onClick={() => handlePullModel(tag)}
                            className="flex items-center gap-1.5 px-3 py-1 text-xs rounded-lg bg-white/10 hover:bg-white/20 text-slate-200 border border-white/10 transition-colors disabled:opacity-50"
                          >
                            <Download size={12} />
                            Download
                          </button>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          ))}

          {/* Pull Custom Tag Input */}
          <div className="bg-white/5 border border-white/10 rounded-xl p-4 space-y-2 font-sans">
            <h4 className="text-xs font-medium text-slate-300">Pull Custom Model Tag</h4>
            <form onSubmit={handleCustomPull} className="flex gap-2">
              <input
                type="text"
                placeholder="Enter model tag to download..."
                value={customTagInput}
                onChange={(e) => setCustomTagInput(e.target.value)}
                className="flex-1 liquid-glass-input rounded-xl px-3 py-1.5 text-xs text-slate-200 placeholder-slate-500 font-mono focus:outline-none"
              />
              <button
                type="submit"
                disabled={!customTagInput.trim() || !!downloadingTag}
                className="px-4 py-1.5 bg-white text-slate-950 hover:bg-slate-200 font-semibold text-xs rounded-xl transition-colors disabled:opacity-50"
              >
                Pull Tag
              </button>
            </form>
          </div>
        </div>

        {/* Footer */}
        <div className="px-6 py-3 border-t border-white/10 bg-white/5 flex items-center justify-between text-xs text-slate-400">
          <span>Ollama Endpoint: <code className="text-slate-300 font-mono">http://localhost:11434</code></span>
          <button
            onClick={onClose}
            className="px-4 py-1.5 rounded-xl bg-white/10 hover:bg-white/20 text-slate-200 transition-colors"
          >
            Done
          </button>
        </div>
      </div>
    </div>
  );
}
