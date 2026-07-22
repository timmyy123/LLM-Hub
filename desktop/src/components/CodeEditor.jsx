import React, { useState, useEffect } from 'react';
import { Save, FileCode, Check, Copy } from 'lucide-react';

export default function CodeEditor({ activeFile, onSaveFile }) {
  const [content, setContent] = useState('');
  const [loading, setLoading] = useState(false);
  const [isSaved, setIsSaved] = useState(true);
  const [saveStatusText, setSaveStatusText] = useState('');
  const [copied, setCopied] = useState(false);

  const filePath = typeof activeFile === 'string' ? activeFile : activeFile?.path;
  const fileName = typeof activeFile === 'string'
    ? activeFile.split(/[\/\\]/).pop()
    : activeFile?.name || (filePath ? filePath.split(/[\/\\]/).pop() : '');

  useEffect(() => {
    if (filePath) {
      setLoading(true);
      if (window.api && window.api.readFile) {
        window.api.readFile(filePath).then((res) => {
          setLoading(false);
          if (res.success && typeof res.content === 'string') {
            setContent(res.content);
            setIsSaved(true);
          } else {
            setContent(`// Error reading file: ${res.error || 'Unknown'}`);
          }
        }).catch((err) => {
          setLoading(false);
          setContent(`// Error loading file: ${err.message}`);
        });
      } else {
        setLoading(false);
      }
    } else {
      setContent('');
    }
  }, [filePath]);

  const handleChange = (e) => {
    setContent(e.target.value);
    setIsSaved(false);
  };

  const handleKeyDown = (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === 's') {
      e.preventDefault();
      handleSave();
    }
    if (e.key === 'Tab') {
      e.preventDefault();
      const start = e.target.selectionStart;
      const end = e.target.selectionEnd;
      const val = content;
      setContent(val.substring(0, start) + '  ' + val.substring(end));
      setTimeout(() => {
        e.target.selectionStart = e.target.selectionEnd = start + 2;
      }, 0);
    }
  };

  const handleSave = async () => {
    if (!filePath) return;
    if (window.api && window.api.writeFile) {
      const res = await window.api.writeFile(filePath, content);
      if (res.success) {
        setIsSaved(true);
        setSaveStatusText('Saved!');
        setTimeout(() => setSaveStatusText(''), 2000);
        if (onSaveFile) onSaveFile(filePath);
      }
    }
  };

  const handleCopy = () => {
    navigator.clipboard.writeText(content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  if (!filePath) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center bg-[#0A0C10] text-slate-500 text-xs font-sans">
        <FileCode size={36} className="mb-2 text-slate-600" />
        <p>Select a file in the workspace explorer to view or edit code</p>
      </div>
    );
  }

  const lines = content ? content.split('\n') : [''];
  const lineCount = lines.length;

  return (
    <div className="flex-1 flex flex-col h-full bg-[#0D0E12] text-slate-200 overflow-hidden select-none font-mono">
      {/* VS Code Editor Tab Bar */}
      <div className="h-10 border-b border-white/10 px-4 bg-black/40 flex items-center justify-between text-xs">
        <div className="flex items-center gap-2">
          <FileCode size={14} className="text-amber-400" />
          <span className="font-semibold text-slate-200">{fileName}</span>
          <span className="text-[10px] text-slate-500 truncate max-w-xs">{filePath}</span>
          {!isSaved && <span className="w-2 h-2 rounded-full bg-amber-400" title="Unsaved changes" />}
        </div>

        <div className="flex items-center gap-2">
          {saveStatusText && (
            <span className="text-emerald-400 text-xs flex items-center gap-1 font-mono">
              <Check size={12} />
              {saveStatusText}
            </span>
          )}

          <button
            onClick={handleCopy}
            className="p-1.5 rounded-lg bg-white/5 hover:bg-white/10 text-slate-400 hover:text-white transition-colors"
            title="Copy Code"
          >
            {copied ? <Check size={13} className="text-emerald-400" /> : <Copy size={13} />}
          </button>

          <button
            onClick={handleSave}
            disabled={isSaved}
            className="flex items-center gap-1.5 px-3 py-1 rounded-lg bg-white/10 hover:bg-white/20 text-slate-200 border border-white/10 font-sans text-xs disabled:opacity-40 transition-colors"
          >
            <Save size={13} />
            <span>Save</span>
          </button>
        </div>
      </div>

      {/* Editor Body */}
      <div className="flex-1 flex overflow-hidden relative">
        {loading ? (
          <div className="absolute inset-0 flex items-center justify-center bg-[#0D0E12] text-slate-400 text-xs">
            Loading file content...
          </div>
        ) : (
          <div className="flex-1 flex overflow-auto custom-scrollbar">
            {/* Line Numbers Gutter */}
            <div className="w-12 bg-black/30 border-r border-white/5 py-3 text-right pr-3 text-slate-600 text-xs font-mono select-none leading-relaxed">
              {Array.from({ length: Math.max(lineCount, 1) }).map((_, i) => (
                <div key={i + 1}>{i + 1}</div>
              ))}
            </div>

            {/* Editable Code Area */}
            <textarea
              value={content}
              onChange={handleChange}
              onKeyDown={handleKeyDown}
              spellCheck={false}
              className="flex-1 bg-transparent text-slate-200 text-xs p-3 focus:outline-none resize-none font-mono leading-relaxed whitespace-pre"
            />
          </div>
        )}
      </div>
    </div>
  );
}
