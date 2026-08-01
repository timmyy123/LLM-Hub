import { useEffect, useState } from 'react';
import { Icon } from './Icon';

interface ModelDownloaderModalProps {
  isOpen: boolean;
  onClose: () => void;
}

interface InstalledModel {
  name: string;
  size: number;
  modified_at: string;
}

const FEATURED_MODELS = [
  {
    family: 'Gemma 4',
    provider: 'Google DeepMind',
    tag: 'gemma4:latest',
    description: 'Multimodal Gemma 4 series optimized for code & vision',
    tags: ['gemma4:latest', 'gemma4:26b', 'gemma4:e4b', 'gemma4:e2b'],
  },
  {
    family: 'Ministral 3',
    provider: 'Mistral AI',
    tag: 'ministral-3:latest',
    description: 'Edge-optimized fast reasoning and agentic model family',
    tags: ['ministral-3:latest', 'ministral-3:8b', 'ministral-3:3b', 'ministral-3:14b'],
  },
  {
    family: 'LFM2 24B',
    provider: 'Liquid AI',
    tag: 'lfm2:24b-a2b',
    description: 'Liquid AI Hybrid SSM-Transformer 24B architecture',
    tags: ['lfm2:24b-a2b', 'lfm2:latest'],
  },
  {
    family: 'Llama 3',
    provider: 'Meta AI',
    tag: 'llama3:latest',
    description: 'General purpose 8B open model for fast local inference',
    tags: ['llama3:latest', 'llama3:70b'],
  },
  {
    family: 'Qwen 2.5 Coder',
    provider: 'Alibaba',
    tag: 'qwen2.5-coder:latest',
    description: 'State-of-the-art open code generation & editing model',
    tags: ['qwen2.5-coder:latest', 'qwen2.5-coder:7b', 'qwen2.5-coder:14b'],
  },
];

export function ModelDownloaderModal({ isOpen, onClose }: ModelDownloaderModalProps) {
  const [installedModels, setInstalledModels] = useState<InstalledModel[]>([]);
  const [customTag, setCustomTag] = useState('');
  const [pullingTag, setPullingTag] = useState<string | null>(null);
  const [pullProgress, setPullProgress] = useState<string>('');
  const [statusMessage, setStatusMessage] = useState<string | null>(null);

  const fetchInstalledModels = async () => {
    try {
      const res = await fetch('http://localhost:11434/api/tags');
      if (res.ok) {
        const data = await res.json();
        setInstalledModels(data.models || []);
      }
    } catch {
      setStatusMessage('Ollama service offline or not responding on http://localhost:11434');
    }
  };

  useEffect(() => {
    if (isOpen) {
      void fetchInstalledModels();
      setStatusMessage(null);
    }
  }, [isOpen]);

  if (!isOpen) return null;

  const handlePull = async (modelTag: string) => {
    if (!modelTag.trim()) return;
    const tag = modelTag.trim();
    setPullingTag(tag);
    setPullProgress('Starting download...');
    setStatusMessage(null);

    try {
      const res = await fetch('http://localhost:11434/api/pull', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: tag, stream: true }),
      });

      if (!res.ok || !res.body) {
        throw new Error(`Failed to pull model ${tag}`);
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const text = decoder.decode(value);
        const lines = text.split('\n').filter(Boolean);
        for (const line of lines) {
          try {
            const parsed = JSON.parse(line);
            if (parsed.status) {
              if (parsed.total && parsed.completed) {
                const pct = Math.round((parsed.completed / parsed.total) * 100);
                setPullProgress(`${parsed.status} (${pct}%)`);
              } else {
                setPullProgress(parsed.status);
              }
            }
          } catch {
            // ignore non-json chunk lines
          }
        }
      }

      setStatusMessage(`Successfully downloaded ${tag}!`);
      void fetchInstalledModels();
    } catch (err) {
      setStatusMessage(`Error downloading ${tag}: ${err instanceof Error ? err.message : 'Unknown error'}`);
    } finally {
      setPullingTag(null);
      setPullProgress('');
    }
  };

  return (
    <div
      className="dialog-backdrop"
      style={{
        position: 'fixed',
        inset: 0,
        backgroundColor: 'rgba(0, 0, 0, 0.7)',
        backdropFilter: 'blur(8px)',
        zIndex: 9999,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '24px',
      }}
      onClick={onClose}
    >
      <div
        className="dialog-content"
        style={{
          width: '100%',
          maxWidth: '720px',
          maxHeight: '85vh',
          backgroundColor: 'var(--color-bg-subtle, #18181b)',
          border: '1px solid var(--color-border-subtle, #27272a)',
          borderRadius: '16px',
          display: 'flex',
          flexDirection: 'column',
          boxShadow: '0 25px 50px -12px rgba(0,0,0,0.5)',
          overflow: 'hidden',
          color: 'var(--color-text, #f4f4f5)',
        }}
        onClick={(e) => e.stopPropagation()}
      >
        <header
          style={{
            padding: '20px 24px',
            borderBottom: '1px solid var(--color-border-subtle, #27272a)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <div
              style={{
                width: '36px',
                height: '36px',
                borderRadius: '10px',
                backgroundColor: 'rgba(255, 255, 255, 0.08)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
            >
              <Icon name="download" size={20} />
            </div>
            <div>
              <h2 style={{ margin: 0, fontSize: '18px', fontWeight: 600 }}>Local Model Downloader</h2>
              <p style={{ margin: '2px 0 0', fontSize: '13px', color: '#a1a1aa' }}>
                Download & manage Ollama models directly on your machine
              </p>
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            style={{
              background: 'none',
              border: 'none',
              color: '#a1a1aa',
              cursor: 'pointer',
              padding: '8px',
              borderRadius: '8px',
            }}
          >
            <Icon name="x" size={18} />
          </button>
        </header>

        <div style={{ padding: '24px', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '24px' }}>
          {statusMessage && (
            <div
              style={{
                padding: '12px 16px',
                borderRadius: '8px',
                backgroundColor: statusMessage.startsWith('Error')
                  ? 'rgba(239, 68, 68, 0.15)'
                  : 'rgba(34, 197, 94, 0.15)',
                border: statusMessage.startsWith('Error')
                  ? '1px solid rgba(239, 68, 68, 0.3)'
                  : '1px solid rgba(34, 197, 94, 0.3)',
                fontSize: '13px',
              }}
            >
              {statusMessage}
            </div>
          )}

          {/* Custom Download Input */}
          <div style={{ display: 'flex', gap: '10px' }}>
            <input
              type="text"
              placeholder="Enter model tag (e.g. gemma4:latest, ministral-3:8b, lfm2:24b-a2b)..."
              value={customTag}
              onChange={(e) => setCustomTag(e.target.value)}
              style={{
                flex: 1,
                padding: '10px 14px',
                borderRadius: '8px',
                backgroundColor: 'rgba(0,0,0,0.2)',
                border: '1px solid var(--color-border-subtle, #3f3f46)',
                color: '#fff',
                fontSize: '14px',
              }}
              onKeyDown={(e) => {
                if (e.key === 'Enter') void handlePull(customTag);
              }}
            />
            <button
              type="button"
              disabled={!customTag.trim() || !!pullingTag}
              onClick={() => void handlePull(customTag)}
              style={{
                padding: '10px 20px',
                borderRadius: '8px',
                backgroundColor: 'var(--color-accent, #2563eb)',
                color: '#fff',
                border: 'none',
                fontWeight: 600,
                cursor: customTag.trim() && !pullingTag ? 'pointer' : 'not-allowed',
                opacity: customTag.trim() && !pullingTag ? 1 : 0.5,
              }}
            >
              {pullingTag === customTag ? 'Downloading...' : 'Pull Model'}
            </button>
          </div>

          {pullingTag && (
            <div
              style={{
                padding: '14px 18px',
                borderRadius: '10px',
                backgroundColor: 'rgba(37, 99, 235, 0.15)',
                border: '1px solid rgba(37, 99, 235, 0.3)',
                display: 'flex',
                alignItems: 'center',
                gap: '12px',
              }}
            >
              <Icon name="loader" size={18} className="animate-spin" />
              <div style={{ flex: 1 }}>
                <div style={{ fontWeight: 600, fontSize: '14px' }}>Downloading {pullingTag}</div>
                <div style={{ fontSize: '13px', color: '#93c5fd', marginTop: '2px' }}>{pullProgress}</div>
              </div>
            </div>
          )}

          {/* Featured Models Section */}
          <div>
            <h3 style={{ fontSize: '14px', textTransform: 'uppercase', letterSpacing: '0.05em', color: '#a1a1aa', margin: '0 0 12px' }}>
              Featured Local Families
            </h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
              {FEATURED_MODELS.map((m) => {
                const isInstalled = installedModels.some((inst) => inst.name === m.tag);
                return (
                  <div
                    key={m.family}
                    style={{
                      padding: '16px',
                      borderRadius: '12px',
                      backgroundColor: 'rgba(255,255,255,0.03)',
                      border: '1px solid rgba(255,255,255,0.08)',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      gap: '16px',
                    }}
                  >
                    <div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <span style={{ fontWeight: 600, fontSize: '15px' }}>{m.family}</span>
                        <span style={{ fontSize: '11px', padding: '2px 8px', borderRadius: '12px', backgroundColor: 'rgba(255,255,255,0.1)', color: '#a1a1aa' }}>
                          {m.provider}
                        </span>
                      </div>
                      <p style={{ margin: '4px 0 8px', fontSize: '13px', color: '#a1a1aa' }}>{m.description}</p>
                      <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap' }}>
                        {m.tags.map((t) => (
                          <button
                            key={t}
                            type="button"
                            onClick={() => void handlePull(t)}
                            style={{
                              padding: '2px 8px',
                              borderRadius: '6px',
                              fontSize: '12px',
                              backgroundColor: 'rgba(255,255,255,0.06)',
                              border: '1px solid rgba(255,255,255,0.1)',
                              color: '#d4d4d8',
                              cursor: 'pointer',
                            }}
                          >
                            {t}
                          </button>
                        ))}
                      </div>
                    </div>
                    <button
                      type="button"
                      disabled={!!pullingTag}
                      onClick={() => void handlePull(m.tag)}
                      style={{
                        padding: '8px 16px',
                        borderRadius: '8px',
                        backgroundColor: isInstalled ? 'rgba(34, 197, 94, 0.2)' : 'var(--color-accent, #2563eb)',
                        color: isInstalled ? '#4ade80' : '#fff',
                        border: isInstalled ? '1px solid rgba(34, 197, 94, 0.4)' : 'none',
                        fontSize: '13px',
                        fontWeight: 600,
                        cursor: pullingTag ? 'not-allowed' : 'pointer',
                        whiteSpace: 'nowrap',
                      }}
                    >
                      {isInstalled ? 'Installed ✓' : 'Download'}
                    </button>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Installed Models Section */}
          <div>
            <h3 style={{ fontSize: '14px', textTransform: 'uppercase', letterSpacing: '0.05em', color: '#a1a1aa', margin: '0 0 12px' }}>
              Installed Models ({installedModels.length})
            </h3>
            {installedModels.length === 0 ? (
              <div style={{ fontSize: '13px', color: '#71717a', fontStyle: 'italic' }}>
                No models installed yet on http://localhost:11434
              </div>
            ) : (
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: '8px' }}>
                {installedModels.map((m) => (
                  <div
                    key={m.name}
                    style={{
                      padding: '10px 14px',
                      borderRadius: '8px',
                      backgroundColor: 'rgba(255,255,255,0.04)',
                      border: '1px solid rgba(255,255,255,0.08)',
                      fontSize: '13px',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                    }}
                  >
                    <span style={{ fontWeight: 500, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {m.name}
                    </span>
                    <span style={{ fontSize: '11px', color: '#4ade80', marginLeft: '6px' }}>✓</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
