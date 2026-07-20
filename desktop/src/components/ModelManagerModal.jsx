import React, { useState, useEffect } from 'react';
import { X, Download, CheckCircle, RefreshCw, AlertTriangle, ShieldCheck, Cpu, HardDrive } from 'lucide-react';
import { ALLOWED_FAMILIES, isAllowedModelTag, filterAllowedModels } from '../services/ollamaService';

export default function ModelManagerModal({ isOpen, onClose, installedModels, selectedModel, onSelectModel, onRefresh }) {
  const [downloadingTag, setDownloadingTag] = useState(null);
  const [downloadProgress, setDownloadProgress] = useState(null);
  const [customTagInput, setCustomTagInput] = useState('');
  const [errorMessage, setErrorMessage] = useState('');

  if (!isOpen) return null;

  const handlePullModel = async (tagToPull) => {
    setErrorMessage('');
    if (!isAllowedModelTag(tagToPull)) {
      setErrorMessage(`Error: Model "${tagToPull}" is not permitted. Only Gemma 4 family, Ministral 3 family, and LFM2 24B A4B models are allowed.`);
      return;
    }

    setDownloadingTag(tagToPull);
    setDownloadProgress({ status: 'Starting download...', percent: 0 });

    if (window.api && window.api.pullModel) {
      const unsub = window.api.onPullProgress((data) => {
        if (data.total && data.completed) {
          const pct = Math.round((data.completed / data.total) * 100);
          setDownloadProgress({ status: data.status, percent: pct, completed: data.completed, total: data.total });
        } else {
          setDownloadProgress({ status: data.status || 'Downloading...', percent: 0 });
        }
      });

      const res = await window.api.pullModel(tagToPull);
      unsub();

      if (res.success) {
        setDownloadProgress({ status: 'Completed!', percent: 100 });
        setTimeout(() => {
          setDownloadingTag(null);
          setDownloadProgress(null);
          onRefresh();
        }, 1200);
      } else {
        setErrorMessage(res.error || 'Download failed');
        setDownloadingTag(null);
        setDownloadProgress(null);
      }
    } else {
      // Mock progress for browser mode
      let p = 0;
      const interval = setInterval(() => {
        p += 20;
        setDownloadProgress({ status: `Downloading layers (${p}%)...`, percent: p });
        if (p >= 100) {
          clearInterval(interval);
          setDownloadingTag(null);
          setDownloadProgress(null);
          onRefresh();
        }
      }, 500);
    }
  };

  const handleCustomPull = (e) => {
    e.preventDefault();
    if (!customTagInput.trim()) return;
    handlePullModel(customTagInput.trim());
    setCustomTagInput('');
  };

  const isInstalled = (tag) => {
    return installedModels.some(m => m.name === tag || m.model === tag || m.name?.startsWith(tag));
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-md p-4 animate-in fade-in duration-200">
      <div className="bg-[#14181F] border border-slate-700/60 rounded-2xl w-full max-w-3xl overflow-hidden shadow-2xl flex flex-col max-h-[85vh]">
        {/* Modal Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-800 bg-[#191E27]">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-amber-500/10 border border-amber-500/30 flex items-center justify-center text-amber-500">
              <Cpu size={20} />
            </div>
            <div>
              <h2 className="text-lg font-semibold text-slate-100 flex items-center gap-2">
                Ollama Local Model Downloader
                <span className="text-xs px-2 py-0.5 rounded-md bg-amber-500/20 text-amber-400 border border-amber-500/30 font-mono">
                  Strict Mode
                </span>
              </h2>
              <p className="text-xs text-slate-400">Exclusive support for requested local model families</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="text-slate-400 hover:text-slate-200 p-1.5 rounded-lg hover:bg-slate-800 transition-colors"
          >
            <X size={20} />
          </button>
        </div>

        {/* Security / Restriction Warning Banner */}
        <div className="px-6 py-2.5 bg-slate-900/60 border-b border-slate-800 flex items-center justify-between text-xs text-slate-300">
          <span className="flex items-center gap-2 text-amber-400">
            <ShieldCheck size={16} />
            Restricted to: Gemma 4 Family, Ministral 3 Family, & LFM2 24B A4B
          </span>
          <button
            onClick={onRefresh}
            className="flex items-center gap-1.5 text-slate-400 hover:text-amber-400 transition-colors"
          >
            <RefreshCw size={13} />
            Refresh Installed
          </button>
        </div>

        {/* Error message */}
        {errorMessage && (
          <div className="mx-6 mt-4 p-3 rounded-xl bg-red-500/10 border border-red-500/30 text-red-400 text-xs flex items-center gap-2">
            <AlertTriangle size={16} className="shrink-0" />
            <span>{errorMessage}</span>
          </div>
        )}

        {/* Pull Progress indicator */}
        {downloadingTag && downloadProgress && (
          <div className="mx-6 mt-4 p-4 rounded-xl bg-amber-500/10 border border-amber-500/30 space-y-2">
            <div className="flex justify-between text-xs font-mono text-amber-300">
              <span className="flex items-center gap-2">
                <RefreshCw size={14} className="animate-spin" />
                Pulling {downloadingTag}...
              </span>
              <span>{downloadProgress.percent}%</span>
            </div>
            <div className="w-full bg-slate-900 h-2 rounded-full overflow-hidden">
              <div
                className="bg-amber-500 h-full transition-all duration-300 rounded-full"
                style={{ width: `${downloadProgress.percent}%` }}
              />
            </div>
            <p className="text-[11px] text-slate-400 font-mono">{downloadProgress.status}</p>
          </div>
        )}

        {/* Modal Body - Model Families List */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {Object.values(ALLOWED_FAMILIES).map((family) => (
            <div key={family.id} className="bg-slate-900/50 border border-slate-800 rounded-xl p-4 space-y-3">
              <div className="flex items-start justify-between">
                <div>
                  <h3 className="text-sm font-semibold text-amber-400 flex items-center gap-2">
                    <HardDrive size={15} />
                    {family.name}
                  </h3>
                  <p className="text-xs text-slate-400 mt-0.5">{family.description}</p>
                </div>
              </div>

              {/* Tags grid */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 pt-1">
                {family.popularTags.map((tag) => {
                  const installed = isInstalled(tag);
                  const isSelected = selectedModel === tag;
                  const isPullingThis = downloadingTag === tag;

                  return (
                    <div
                      key={tag}
                      className={`flex items-center justify-between p-2.5 rounded-lg border transition-all ${
                        isSelected
                          ? 'bg-amber-500/15 border-amber-500/60 text-slate-100'
                          : 'bg-slate-800/40 border-slate-800 text-slate-300 hover:border-slate-700'
                      }`}
                    >
                      <div className="flex items-center gap-2 min-w-0">
                        <span className="font-mono text-xs font-medium truncate">{tag}</span>
                        {installed && (
                          <span className="text-[10px] px-1.5 py-0.5 rounded bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
                            Installed
                          </span>
                        )}
                      </div>

                      <div className="flex items-center gap-2 shrink-0">
                        {installed ? (
                          <button
                            onClick={() => onSelectModel(tag)}
                            className={`px-2.5 py-1 text-xs rounded-md font-medium transition-colors ${
                              isSelected
                                ? 'bg-amber-500 text-slate-950 hover:bg-amber-400'
                                : 'bg-slate-700 text-slate-200 hover:bg-slate-600'
                            }`}
                          >
                            {isSelected ? 'Active' : 'Select'}
                          </button>
                        ) : (
                          <button
                            disabled={isPullingThis || !!downloadingTag}
                            onClick={() => handlePullModel(tag)}
                            className="flex items-center gap-1.5 px-2.5 py-1 text-xs rounded-md bg-amber-500/20 text-amber-300 hover:bg-amber-500/30 border border-amber-500/30 transition-colors disabled:opacity-50"
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

          {/* Custom Tag pull input matching allowed families */}
          <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-4 space-y-2">
            <h4 className="text-xs font-medium text-slate-300">Pull Custom Allowed Tag</h4>
            <form onSubmit={handleCustomPull} className="flex gap-2">
              <input
                type="text"
                placeholder="e.g. gemma4:9b or ministral-3:8b or lfm2:24b-a4b"
                value={customTagInput}
                onChange={(e) => setCustomTagInput(e.target.value)}
                className="flex-1 bg-slate-950 border border-slate-800 rounded-lg px-3 py-1.5 text-xs text-slate-200 placeholder-slate-500 font-mono focus:outline-none focus:border-amber-500/60"
              />
              <button
                type="submit"
                disabled={!customTagInput.trim() || !!downloadingTag}
                className="px-4 py-1.5 bg-amber-500 hover:bg-amber-400 text-slate-950 font-medium text-xs rounded-lg transition-colors disabled:opacity-50"
              >
                Pull Tag
              </button>
            </form>
          </div>
        </div>

        {/* Modal Footer */}
        <div className="px-6 py-3 border-t border-slate-800 bg-[#191E27] flex items-center justify-between text-xs text-slate-400">
          <span>Ollama API Endpoint: <code className="text-slate-300">http://localhost:11434</code></span>
          <button
            onClick={onClose}
            className="px-4 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
